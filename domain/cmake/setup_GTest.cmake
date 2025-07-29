#####################################################
# GTest, find system first, download if necessary
#####################################################
find_package(GTest)

if (NOT GTest_FOUND)
    message(STATUS "Configure GTest from github")

    include(FetchContent)
    FetchContent_Declare(
      googletest
      GIT_REPOSITORY https://github.com/google/googletest.git
      GIT_TAG        v1.13.0
    )

    # Check if population has already been performed
    FetchContent_GetProperties(googletest)
    if(NOT googletest_POPULATED)
        message(STATUS "Downloading GTest from github")
        # Fetch the content using previously declared details
        FetchContent_MakeAvailable(googletest)

        # Prevent overriding the parent project's compiler/linker
        # settings on Windows
        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    endif()
endif()

if (NOT TARGET GTest::gtest_main)
    message("Target GTest:: stuff MISSING")
endif()
