# ------------------------------------------------------------------------------
# H5hut
# Use find_package(...) with a mininimum version requested, except:
#   - if the user has requested a version that starts with "git.(tag/branch/sha)"
#   - if the requested version is not found on the system
# then build from source.
# we use FIND_PACKAGE_ARGS to allow FetchContent to find a system version
# ------------------------------------------------------------------------------
include(FetchContent)

# ------------------------------------------------------------------------------
# Utility function to get git tag/sha/version from version string
# ------------------------------------------------------------------------------
function(extract_git_label VERSION_STRING RESULT_VAR)
  if("${${VERSION_STRING}}" MATCHES "^git\\.(.+)$")
    set(${RESULT_VAR} "${CMAKE_MATCH_1}" PARENT_SCOPE)
  else()
    unset(${RESULT_VAR} PARENT_SCOPE)
  endif()
endfunction()

# ------------------------------------------------------------------------------
# set the default version of kokkos we will ask for if not already set
if (NOT H5hut_VERSION_DEFAULT)
  set(H5hut_VERSION_DEFAULT 2.0)
endif()
# if the user has not asked for a specific version, we will use a default
if (NOT H5hut_VERSION)
  set(H5hut_VERSION ${H5hut_VERSION_DEFAULT})
endif()

# is H5hut_VERSION a git tag/branch/sha? if it is, we check it out,
# otherwise, we use FIND_PACKAGE_ARGS (cmake version 3.24+) to allow FetchContent 
# to find a system version before downloading from source
extract_git_label(H5hut_VERSION H5HUT_VERSION_GIT)
if (H5HUT_VERSION_GIT)
  # the user has asked for a particular version build from source
  set(fetch_string
    GIT_TAG ${H5HUT_VERSION_GIT}
    GIT_REPOSITORY https://github.com/eth-cscs/h5hut)
else()
  # the user has asked for a version - use find or checkout if find fails
  set(fetch_string
    GIT_TAG ${H5hut_VERSION}
    GIT_REPOSITORY https://github.com/eth-cscs/h5hut
    FIND_PACKAGE_ARGS ${H5hut_VERSION} COMPONENTS ${IPPL_PLATFORMS}
  )
endif()

# Invoke cmake fetch/find
FetchContent_Declare(H5hut ${fetch_string})
FetchContent_MakeAvailable(H5hut)

# Check that kokkos actually has the platform backends that we need
if (H5hut_FOUND)
  message(STATUS "H5hut ${H5hut_VERSION} found externally")
else()
  message(STATUS "H5hut ${H5hut_VERSION} building from source in ${H5hut_SOURCE_DIR}")
  set(H5hut_FOUND ON)
endif()

