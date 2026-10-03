#include "artifact/reader.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstring>
#include <functional>
#include <limits>
#include <span>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <unordered_map>
#include <utility>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace ninfer::artifact {
namespace {

using Json = nlohmann::json;

// "NINFER\0" followed by one version byte: 2 or 3.
constexpr std::array<std::byte, 7> kMagicStem = {
    std::byte{'N'}, std::byte{'I'}, std::byte{'N'}, std::byte{'F'},
    std::byte{'E'}, std::byte{'R'}, std::byte{0},
};
constexpr std::uint64_t kPrefixBytes      = 16;
constexpr std::uint64_t kPayloadAlignment = 4096;
// v3 puts 16 opaque bytes between the prefix and its JSON directory.
constexpr std::uint64_t kV3DirectoryLead = 16;

std::uint64_t checked_add(std::uint64_t a, std::uint64_t b, std::string_view label) {
    if (b > std::numeric_limits<std::uint64_t>::max() - a) {
        throw ArtifactError(std::string(label) + " overflows u64");
    }
    return a + b;
}

std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment, std::string_view label) {
    const auto biased = checked_add(value, alignment - 1, label);
    return biased / alignment * alignment;
}

std::uint64_t read_u64_le(const std::byte* data) noexcept {
    std::uint64_t value = 0;
    for (unsigned i = 0; i < 8; ++i) {
        value |= std::uint64_t(std::to_integer<unsigned char>(data[i])) << (i * 8);
    }
    return value;
}

template <std::size_t N>
void require_members(const Json& value, const std::array<const char*, N>& members,
                     std::string_view label) {
    if (!value.is_object() || value.size() != N) {
        throw ArtifactError(std::string(label) + " has missing or extra members");
    }
    for (const char* member : members) {
        if (!value.contains(member)) {
            throw ArtifactError(std::string(label) + " has missing or extra members");
        }
    }
}

const std::string& require_string(const Json& value, std::string_view label) {
    if (!value.is_string()) {
        throw ArtifactError(std::string(label) + " must be a nonempty string");
    }
    const auto& result = value.get_ref<const std::string&>();
    if (result.empty()) { throw ArtifactError(std::string(label) + " must be a nonempty string"); }
    return result;
}

std::uint64_t require_unsigned(const Json& value, std::string_view label, bool positive) {
    if (!value.is_number_unsigned()) {
        throw ArtifactError(std::string(label) + " must be an integer");
    }
    const auto result = value.get<std::uint64_t>();
    if (positive && result == 0) { throw ArtifactError(std::string(label) + " must be positive"); }
    return result;
}

// Each name pair is the v2 spelling, then the v3 spelling of the same encoding.
NumericFormat parse_format(std::string_view name) {
    if (name == "BF16" || name == "bf16") { return NumericFormat::BF16; }
    if (name == "FP32" || name == "fp32") { return NumericFormat::FP32; }
    if (name == "I32" || name == "int32") { return NumericFormat::I32; }
    if (name == "Q4G64_F16S" || name == "q4_g64_fp16") { return NumericFormat::Q4G64_F16S; }
    if (name == "Q5G64_F16S" || name == "q5_g64_fp16") { return NumericFormat::Q5G64_F16S; }
    if (name == "Q6G64_F16S" || name == "q6_g64_fp16") { return NumericFormat::Q6G64_F16S; }
    if (name == "W8G32_F16S" || name == "q8_g32_fp16") { return NumericFormat::W8G32_F16S; }
    if (name == "NVFP4" || name == "nvfp4") { return NumericFormat::NVFP4; }
    if (name == "FP8_E4M3FN_ROW_BF16S" || name == "fp8_e4m3fn_row_bf16") {
        return NumericFormat::FP8_E4M3FN_ROW_BF16S;
    }
    throw ArtifactError("unknown tensor format: " + std::string(name));
}

StorageLayout parse_layout(std::string_view name) {
    if (name == "contiguous-le-v1" || name == "contiguous_le_v1") {
        return StorageLayout::ContiguousLeV1;
    }
    if (name == "row-split-k128-v1" || name == "row_split_k128_v1") {
        return StorageLayout::RowSplitK128V1;
    }
    if (name == "blockscale-k16-m128x4-v1" || name == "block_scale_k16_m128x4_v1") {
        return StorageLayout::BlockScaleK16M128x4V1;
    }
    if (name == "row-scale-v1" || name == "row_scale_v1") { return StorageLayout::RowScaleV1; }
    throw ArtifactError("unknown tensor layout: " + std::string(name));
}

ResourceEncoding parse_encoding(std::string_view name) {
    if (name == "raw-bytes-v1" || name == "raw_bytes_v1") { return ResourceEncoding::RawBytesV1; }
    throw ArtifactError("unknown resource encoding: " + std::string(name));
}

TensorDescriptor tensor_from(std::string name, const Json& value) {
    const auto format      = parse_format(require_string(value.at("format"), "tensor format"));
    const auto layout      = parse_layout(require_string(value.at("layout"), "tensor layout"));
    const auto offset      = require_unsigned(value.at("offset"), "tensor offset", false);
    const auto stored_size = require_unsigned(value.at("bytes"), "tensor bytes", true);

    const auto& raw_shape = value.at("shape");
    if (!raw_shape.is_array()) { throw ArtifactError("tensor shape must be an array"); }
    std::vector<std::uint64_t> shape;
    shape.reserve(raw_shape.size());
    for (const auto& dim : raw_shape) {
        shape.push_back(require_unsigned(dim, "shape dimension", true));
    }

    const auto expected_size = tensor_encoded_size(layout, format, shape);
    if (stored_size != expected_size) {
        throw ArtifactError("tensor " + name + " stores " + std::to_string(stored_size) +
                            " bytes; layout requires " + std::to_string(expected_size));
    }
    return {std::move(name), std::move(shape), format, layout, offset, stored_size};
}

TensorDescriptor parse_tensor(const Json& value) {
    static constexpr std::array members = {
        "name", "kind", "shape", "format", "layout", "offset", "bytes",
    };
    require_members(value, members, "tensor entry");
    return tensor_from(require_string(value.at("name"), "tensor name"), value);
}

ResourceDescriptor parse_resource(const Json& value) {
    static constexpr std::array members = {
        "name", "kind", "encoding", "offset", "bytes",
    };
    require_members(value, members, "resource entry");
    return {
        require_string(value.at("name"), "resource name"),
        parse_encoding(require_string(value.at("encoding"), "resource encoding")),
        require_unsigned(value.at("offset"), "resource offset", false),
        require_unsigned(value.at("bytes"), "resource bytes", true),
    };
}

ObjectDescriptor parse_object(const Json& value) {
    if (!value.is_object()) { throw ArtifactError("each object entry must be a JSON object"); }
    const auto it = value.find("kind");
    if (it == value.end() || !it->is_string()) {
        throw ArtifactError("object kind must be 'tensor' or 'resource'");
    }
    const auto& kind = it->get_ref<const std::string&>();
    if (kind == "tensor") { return parse_tensor(value); }
    if (kind == "resource") { return parse_resource(value); }
    throw ArtifactError("object kind must be 'tensor' or 'resource'");
}

struct Directory {
    ArtifactIdentity identity;
    std::vector<ObjectDescriptor> objects;
};

Directory parse_v2(const Json& directory) {
    static constexpr std::array root_members = {"identity", "objects"};
    require_members(directory, root_members, "directory root");
    const auto& raw_identity                     = directory.at("identity");
    static constexpr std::array identity_members = {"model_id", "weights_id"};
    require_members(raw_identity, identity_members, "artifact identity");

    Directory result;
    result.identity.model_id   = require_string(raw_identity.at("model_id"), "model_id");
    result.identity.weights_id = require_string(raw_identity.at("weights_id"), "weights_id");
    const auto& raw_objects    = directory.at("objects");
    if (!raw_objects.is_array() || raw_objects.empty()) {
        throw ArtifactError("objects must be a nonempty array");
    }
    result.objects.reserve(raw_objects.size());
    for (const auto& raw_object : raw_objects) { result.objects.push_back(parse_object(raw_object)); }
    return result;
}

// v3 names objects by opaque id and exposes logical names through bindings. The targets
// look tensors up by their v2 names, so v3 bindings are translated back to those names.
std::string v2_name(std::string name) {
    if (name == "proposal/head") { return "text/draft_head"; }
    if (name == "proposal/token_ids") { return "text/draft_head_token_ids"; }
    constexpr std::string_view mtp_layer = "mtp/layers/0/";
    if (name.starts_with(mtp_layer)) { name = "mtp/layer/" + name.substr(mtp_layer.size()); }
    // ".../norm1_weight" -> ".../norm1/weight"
    const auto leaf_begin = name.rfind('/') + 1;
    const std::string_view leaf(name.data() + leaf_begin, name.size() - leaf_begin);
    const auto underscore = leaf.find('_');
    if (leaf.starts_with("norm") && underscore != std::string_view::npos &&
        (leaf.ends_with("_weight") || leaf.ends_with("_bias")) &&
        std::all_of(leaf.begin() + 4, leaf.begin() + underscore,
                    [](char c) { return c >= '0' && c <= '9'; })) {
        name[leaf_begin + underscore] = '/';
    }
    return name;
}

// v2 fused tensor leaf for the v3 row views that cover one object, joined in row order.
std::string_view fused_leaf(std::string_view dir, std::string_view parts) {
    if (parts == "query,key,value,z") { return "query_key_value_z"; }
    if (parts == "a_projection,b_projection") { return "a_b_projection"; }
    if (parts == "gate,up") { return "gate_up"; }
    if (parts == "query,key,gate,value") { return "query_key_gate_value"; }
    if (parts == "query_bias,key_bias,value_bias") { return "qkv_bias"; }
    if (parts == "query,key,value") {
        return dir.starts_with("vision/") ? "qkv" : "query_key_value";
    }
    return {};
}

// ponytail: weights_id from the recipe suffix; covers the nvfp4 and groupwise-int targets
// this runtime knows, extend when a v3 recipe names another weight scheme.
std::string v2_weights_id(std::string_view recipe) {
    return recipe.ends_with("_nvfp4") ? "nvfp4" : "groupwise-int";
}

Directory parse_v3(const Json& directory, std::span<const std::byte> payload) {
    for (const char* member : {"metadata", "provenance", "objects", "bindings", "uses", "files"}) {
        if (!directory.contains(member)) {
            throw ArtifactError(std::string("v3 directory is missing ") + member);
        }
    }
    const auto& files = directory.at("files");
    if (!files.is_array() || files.size() != 1 || !files[0].at("path").is_null() ||
        require_unsigned(files[0].at("payload_bytes"), "v3 payload_bytes", true) != payload.size()) {
        throw ArtifactError("only single-file v3 artifacts with an exact payload are supported");
    }

    Directory result;
    result.identity.model_id = require_string(directory.at("metadata").at("name"), "v3 model name");
    result.identity.weights_id =
        v2_weights_id(require_string(directory.at("provenance").at("recipe"), "v3 recipe"));

    std::unordered_map<std::string, const Json*> objects;
    for (const auto& object : directory.at("objects")) {
        objects.emplace(require_string(object.at("id"), "v3 object id"), &object);
    }
    std::unordered_map<std::string, std::string> names; // object id -> v2 name
    auto name_object = [&](const std::string& id, std::string name) {
        if (!objects.contains(id)) { throw ArtifactError("v3 binding names unknown object " + id); }
        auto [it, inserted] = names.emplace(id, name);
        if (!inserted && it->second != name) {
            throw ArtifactError("v3 object " + id + " is bound as both " + it->second + " and " +
                                name);
        }
    };

    struct RowView {
        std::uint64_t begin, end;
        std::string leaf, dir;
    };
    std::unordered_map<std::string, std::vector<RowView>> views; // object id -> row views
    for (const auto& [name, binding] : directory.at("bindings").items()) {
        if (binding.contains("object")) {
            name_object(require_string(binding.at("object"), "v3 binding object"), v2_name(name));
            continue;
        }
        const auto& parts = binding.at("parts");
        if (!parts.is_array() || parts.size() != 1) {
            throw ArtifactError("v3 binding " + name + " concatenates objects; unsupported");
        }
        const auto slash = name.rfind('/');
        RowView view{
            require_unsigned(parts[0].at("range")[0], "v3 range", false),
            require_unsigned(parts[0].at("range")[1], "v3 range", true),
            name.substr(slash + 1),
            name.substr(0, slash),
        };
        // dflash2 context_key/context_value alias the key/value rows of the same object.
        if (view.leaf.starts_with("context_")) { continue; }
        views[require_string(parts[0].at("object"), "v3 part object")].push_back(std::move(view));
    }
    for (auto& [id, rows] : views) {
        std::ranges::sort(rows, {}, &RowView::begin);
        const auto it = objects.find(id);
        if (it == objects.end()) { throw ArtifactError("v3 binding names unknown object " + id); }
        std::uint64_t elements = 1;
        for (const auto& dim : it->second->at("shape")) { elements *= dim.get<std::uint64_t>(); }
        std::string joined;
        std::uint64_t cursor = 0;
        for (const auto& row : rows) {
            if (row.begin != cursor || row.dir != rows.front().dir) {
                throw ArtifactError("v3 row views of " + id + " are not one contiguous tensor");
            }
            cursor = row.end;
            joined += (joined.empty() ? "" : ",") + row.leaf;
        }
        const auto leaf = fused_leaf(rows.front().dir, joined);
        if (cursor != elements || leaf.empty()) {
            throw ArtifactError("unsupported v3 row views of " + id + ": " + joined);
        }
        name_object(id, v2_name(rows.front().dir + "/" + std::string(leaf)));
    }

    // FFN activation divisors: v2 keeps one per gate_up pair, v3 one per projection.
    auto object_bytes_of = [&](const std::string& id) {
        const auto& object = *objects.at(id);
        const auto offset  = require_unsigned(object.at("offset"), "v3 offset", false);
        const auto bytes   = require_unsigned(object.at("bytes"), "v3 bytes", true);
        if (offset > payload.size() || bytes > payload.size() - offset) {
            throw ArtifactError("v3 object " + id + " extends beyond the file");
        }
        return payload.subspan(offset, bytes);
    };
    std::unordered_map<std::string, std::string> up_divisors; // mlp dir -> id
    for (const auto& use : directory.at("uses")) {
        if (!use.contains("auxiliaries")) { continue; }
        const auto& auxiliaries = use.at("auxiliaries");
        if (auxiliaries.size() != 1 || !auxiliaries.contains("activation_input_divisor")) {
            throw ArtifactError("unsupported v3 auxiliaries: " + auxiliaries.dump());
        }
        const auto id = require_string(auxiliaries.at("activation_input_divisor").at("object"),
                                       "v3 auxiliary object");
        const auto& parameter = require_string(use.at("parameter"), "v3 use parameter");
        const auto slash      = parameter.rfind('/');
        const auto dir        = parameter.substr(0, slash);
        const auto leaf       = std::string_view(parameter).substr(slash + 1);
        if (leaf == "gate") {
            name_object(id, v2_name(dir + "/gate_up_projection/input_scale_divisor"));
        } else if (leaf == "down") {
            name_object(id, v2_name(dir + "/down_projection/input_scale_divisor"));
        } else if (leaf == "up") {
            up_divisors.emplace(dir, id);
        } else {
            throw ArtifactError("unsupported v3 activation divisor for " + parameter);
        }
    }
    for (const auto& [dir, up_id] : up_divisors) {
        const auto gate = names.end() != std::ranges::find_if(names, [&](const auto& entry) {
            return entry.second == v2_name(dir + "/gate_up_projection/input_scale_divisor") &&
                   std::ranges::equal(object_bytes_of(entry.first), object_bytes_of(up_id));
        });
        if (!gate) {
            throw ArtifactError("v3 gate and up activation divisors differ in " + dir);
        }
    }

    for (const auto& [id, object] : objects) {
        const auto kind = require_string(object->at("kind"), "v3 object kind");
        if (kind == "resource") {
            // "resource/<component>/<file>" -> "frontend/<file>"
            names.emplace(id, "frontend/" + id.substr(id.rfind('/') + 1));
        }
        const auto it = names.find(id);
        if (it == names.end()) {
            if (std::ranges::any_of(up_divisors, [&](const auto& e) { return e.second == id; })) {
                continue;
            }
            throw ArtifactError("v3 object " + id + " has no v2 name");
        }
        if (kind == "tensor") {
            result.objects.push_back(tensor_from(it->second, *object));
        } else if (kind == "resource") {
            result.objects.push_back(ResourceDescriptor{
                it->second,
                parse_encoding(require_string(object->at("encoding"), "resource encoding")),
                require_unsigned(object->at("offset"), "resource offset", false),
                require_unsigned(object->at("bytes"), "resource bytes", true),
            });
        } else {
            throw ArtifactError("v3 object kind must be 'tensor' or 'resource'");
        }
    }
    std::ranges::sort(result.objects, {}, [](const auto& o) { return object_offset(o); });
    return result;
}

struct TransparentStringHash {
    using is_transparent = void;

    std::size_t operator()(std::string_view value) const noexcept {
        return std::hash<std::string_view>{}(value);
    }

    std::size_t operator()(const std::string& value) const noexcept {
        return (*this)(std::string_view(value));
    }
};

class MappedFile {
public:
    explicit MappedFile(const std::filesystem::path& path) {
        const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECT);
        if (fd < 0) {
            throw std::system_error(errno, std::generic_category(), "open " + path.string());
        }

        struct stat status {};

        if (::fstat(fd, &status) != 0) {
            const int error = errno;
            ::close(fd);
            throw std::system_error(error, std::generic_category(), "fstat " + path.string());
        }
        if (status.st_size < 0 ||
            static_cast<std::uintmax_t>(status.st_size) > std::numeric_limits<std::size_t>::max()) {
            ::close(fd);
            throw ArtifactError("artifact size does not fit the process address space");
        }

        const auto size = static_cast<std::size_t>(status.st_size);
        void* mapping   = nullptr;
        if (size != 0) {
            mapping = ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
            if (mapping == MAP_FAILED) {
                const int error = errno;
                ::close(fd);
                throw std::system_error(error, std::generic_category(), "mmap " + path.string());
            }
        }
        fd_   = fd;
        data_ = static_cast<const std::byte*>(mapping);
        size_ = size;
    }

    ~MappedFile() {
        if (data_ != nullptr) { ::munmap(const_cast<std::byte*>(data_), size_); }
        if (fd_ >= 0) { ::close(fd_); }
    }

    MappedFile(const MappedFile&)            = delete;
    MappedFile& operator=(const MappedFile&) = delete;

    const std::byte* data() const noexcept { return data_; }

    std::size_t size() const noexcept { return size_; }

    std::size_t read_direct(std::uint64_t absolute_offset, std::span<std::byte> destination) const {
        constexpr std::size_t alignment = Reader::direct_io_alignment;
        if (absolute_offset % alignment != 0 || destination.size() % alignment != 0 ||
            reinterpret_cast<std::uintptr_t>(destination.data()) % alignment != 0) {
            throw ArtifactError("direct artifact read is not 4096-byte aligned");
        }
        if (absolute_offset > static_cast<std::uint64_t>(std::numeric_limits<off_t>::max()) ||
            destination.size() > static_cast<std::size_t>(std::numeric_limits<ssize_t>::max())) {
            throw ArtifactError("direct artifact read exceeds platform I/O limits");
        }

        ssize_t bytes = -1;
        do {
            bytes = ::pread(fd_, destination.data(), destination.size(),
                            static_cast<off_t>(absolute_offset));
        } while (bytes < 0 && errno == EINTR);
        if (bytes < 0) {
            throw std::system_error(errno, std::generic_category(), "direct artifact read");
        }
        return static_cast<std::size_t>(bytes);
    }

private:
    int fd_                = -1;
    const std::byte* data_ = nullptr;
    std::size_t size_      = 0;
};

} // namespace

std::string_view object_name(const ObjectDescriptor& object) noexcept {
    return std::visit([](const auto& descriptor) -> std::string_view { return descriptor.name; },
                      object);
}

std::uint64_t object_offset(const ObjectDescriptor& object) noexcept {
    return std::visit([](const auto& descriptor) { return descriptor.offset; }, object);
}

std::uint64_t object_bytes(const ObjectDescriptor& object) noexcept {
    return std::visit([](const auto& descriptor) { return descriptor.bytes; }, object);
}

struct Reader::Impl {
    explicit Impl(const std::filesystem::path& path) : file(path) {
        if (file.size() < kPrefixBytes) {
            throw ArtifactError("artifact is shorter than the v2 prefix");
        }
        const auto version = std::to_integer<unsigned>(file.data()[kMagicStem.size()]);
        if (!std::equal(kMagicStem.begin(), kMagicStem.end(), file.data()) ||
            (version != 2 && version != 3)) {
            throw ArtifactError("artifact magic is not NInfer v2 or v3");
        }
        const auto directory_lead = version == 3 ? kV3DirectoryLead : 0;

        const auto json_bytes = read_u64_le(file.data() + 8);
        if (json_bytes <= directory_lead) { throw ArtifactError("json_bytes must be positive"); }
        const auto metadata_end = checked_add(kPrefixBytes, json_bytes, "JSON range");
        payload_start           = align_up(metadata_end, kPayloadAlignment, "payload offset");
        if (metadata_end > file.size() || payload_start > file.size()) {
            throw ArtifactError("declared JSON or payload start extends beyond the file");
        }

        Json directory;
        try {
            const auto* begin =
                reinterpret_cast<const char*>(file.data() + kPrefixBytes + directory_lead);
            directory = Json::parse(begin, begin + (json_bytes - directory_lead));
        } catch (const Json::exception& error) {
            throw ArtifactError(std::string("invalid JSON directory: ") + error.what());
        }

        const auto payload_bytes = static_cast<std::uint64_t>(file.size()) - payload_start;
        auto parsed              = Directory{};
        try {
            parsed = version == 3
                         ? parse_v3(directory, {file.data() + payload_start,
                                                static_cast<std::size_t>(payload_bytes)})
                         : parse_v2(directory);
        } catch (const Json::exception& error) {
            throw ArtifactError(std::string("invalid v3 directory: ") + error.what());
        }
        identity = std::move(parsed.identity);
        entries.reserve(parsed.objects.size());
        index.reserve(parsed.objects.size());

        std::uint64_t cursor = 0;
        for (auto& object : parsed.objects) {
            const auto name      = object_name(object);
            const auto offset    = object_offset(object);
            const auto bytes     = object_bytes(object);
            const auto alignment = std::visit(
                [](const auto& descriptor) {
                    using Descriptor = std::decay_t<decltype(descriptor)>;
                    if constexpr (std::is_same_v<Descriptor, TensorDescriptor>) {
                        return tensor_alignment(descriptor.layout);
                    } else {
                        return resource_alignment(descriptor.encoding);
                    }
                },
                object);

            if (offset < cursor) {
                throw ArtifactError("object " + std::string(name) + " overlaps or is out of order");
            }
            if (offset % alignment != 0) {
                throw ArtifactError("object " + std::string(name) + " is not " +
                                    std::to_string(alignment) + "-byte aligned");
            }
            const auto end = checked_add(offset, bytes, "object payload range");
            if (end > payload_bytes) {
                throw ArtifactError("object " + std::string(name) + " extends beyond the file");
            }
            const auto object_index = entries.size();
            auto [_, inserted]      = index.emplace(std::string(name), object_index);
            if (!inserted) { throw ArtifactError("duplicate object name: " + std::string(name)); }
            entries.push_back(std::move(object));
            cursor = end;
        }
    }

    MappedFile file;
    ArtifactIdentity identity;
    std::vector<ObjectDescriptor> entries;
    std::unordered_map<std::string, std::size_t, TransparentStringHash, std::equal_to<>> index;
    std::uint64_t payload_start = 0;
};

Reader::Reader(const std::filesystem::path& path) : impl_(std::make_unique<Impl>(path)) {}

Reader::~Reader()                            = default;
Reader::Reader(Reader&&) noexcept            = default;
Reader& Reader::operator=(Reader&&) noexcept = default;

const ArtifactIdentity& Reader::identity() const noexcept { return impl_->identity; }

const std::vector<ObjectDescriptor>& Reader::objects() const noexcept { return impl_->entries; }

const ObjectDescriptor* Reader::find(std::string_view name) const noexcept {
    const auto it = impl_->index.find(name);
    return it == impl_->index.end() ? nullptr : &impl_->entries[it->second];
}

std::uint64_t Reader::file_bytes() const noexcept { return impl_->file.size(); }

std::uint64_t Reader::payload_offset() const noexcept { return impl_->payload_start; }

PayloadSpan Reader::payload(const ObjectDescriptor& object) const {
    const auto absolute =
        checked_add(impl_->payload_start, object_offset(object), "absolute payload offset");
    const auto end = checked_add(absolute, object_bytes(object), "absolute payload range");
    if (end > impl_->file.size()) { throw ArtifactError("object payload extends beyond the file"); }
    return {
        absolute,
        std::span<const std::byte>(impl_->file.data() + absolute,
                                   static_cast<std::size_t>(object_bytes(object))),
    };
}

PayloadSpan Reader::payload(std::string_view name) const {
    const auto* object = find(name);
    if (object == nullptr) { throw ArtifactError("unknown artifact object: " + std::string(name)); }
    return payload(*object);
}

std::size_t Reader::read_direct(std::uint64_t absolute_offset,
                                std::span<std::byte> destination) const {
    return impl_->file.read_direct(absolute_offset, destination);
}

} // namespace ninfer::artifact
