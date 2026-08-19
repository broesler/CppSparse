#!/usr/bin/env bash
#===============================================================================
#     File: clang_query_script.sh
#  Created: 2026-08-18 20:45
#   Author: Bernie Roesler
#===============================================================================

jq -r '.[].file' build/Release/compile_commands.json | sort -u | while IFS= read -r f; do
    clang-query -p build/Release \
        -c='match cxxOperatorCallExpr(
            hasAnyArgument(hasType(cxxRecordDecl(hasName("Vector")))),
            anyOf(
                hasOverloadedOperatorName("+"),
                hasOverloadedOperatorName("-"),
                hasOverloadedOperatorName("*"),
                hasOverloadedOperatorName("/"),
                hasOverloadedOperatorName("+="),
                hasOverloadedOperatorName("-="),
                hasOverloadedOperatorName("*="),
                hasOverloadedOperatorName("/=")
            )
        )' \
        "$f"
done

#===============================================================================
#===============================================================================
