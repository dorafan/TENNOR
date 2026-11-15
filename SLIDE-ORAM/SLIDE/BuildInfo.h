#ifndef SLIDE_ORAM_BUILD_INFO_H
#define SLIDE_ORAM_BUILD_INFO_H

#include <cstdio>
#include <cstring>
#include "Config.h"

namespace slide_oram {

// Return -1 only for a normal configuration-file invocation. Keep this handler
// independent of model/data initialization so preflight can query the binary.
inline int handle_build_info_option(int argc, char* argv[]) {
    if (argc == 2 && std::strcmp(argv[1], "--build-info") == 0) {
        // The launcher checks this embedded marker before executing the query,
        // because older binaries treat an unknown option as a training config.
        std::printf("{\"format\":\"SLIDE_ORAM_BUILD_INFO_V1\",\"schema_version\":1,"
                    "\"ORAM_N\":%d,\"SERIAL_LEN\":%d}\n",
                    ORAM_N, SERIAL_LEN);
        return 0;
    }
    if (argc != 2 || argv[1][0] == '-') {
        std::fprintf(stderr, "Usage: %s CONFIG | --build-info\n", argv[0]);
        return 2;
    }
    return -1;
}

}  // namespace slide_oram

#endif
