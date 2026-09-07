function(ryoanji_add_test name)
  set(options FAILURE_EXPECTED RUN_SERIAL)
  set(one_value_args EXECUTABLE RANKS TIMEOUT)
  set(multi_value_args ARGS)
  cmake_parse_arguments(
    ${name} "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN}
  )

  if(NOT ${name}_RANKS)
    set(${name}_RANKS 1)
  endif()

  if(NOT ${name}_EXECUTABLE)
    set(${name}_EXECUTABLE ${name})
  endif()

  if(TARGET ${${name}_EXECUTABLE}_test)
    set(_exe "$<TARGET_FILE:${${name}_EXECUTABLE}_test>")
  elseif(TARGET ${${name}_EXECUTABLE})
    set(_exe "$<TARGET_FILE:${${name}_EXECUTABLE}>")
  else()
    set(_exe "${${name}_EXECUTABLE}")
  endif()

  if(${name}_RUN_SERIAL)
    set(run_serial TRUE)
  endif()

  set(args "${${name}_UNPARSED_ARGUMENTS}" ${args})

  set(_script_location ${PROJECT_BINARY_DIR})

  set(cmd ${_exe})

  list(PREPEND cmd "${MPIEXEC_EXECUTABLE}" "${MPIEXEC_NUMPROC_FLAG}"
     "${${name}_RANKS}" "${MPIEXEC_PREFLAGS}"
  )

  add_test(NAME "${name}" COMMAND ${cmd} ${args})
  if(${run_serial})
    set_tests_properties("${_full_name}" PROPERTIES RUN_SERIAL TRUE)
  endif()
  if(${name}_TIMEOUT)
    set_tests_properties("${_full_name}" PROPERTIES TIMEOUT ${${name}_TIMEOUT})
  endif()
  if(${name}_FAILURE_EXPECTED)
    set_tests_properties("${_full_name}" PROPERTIES WILL_FAIL TRUE)
  endif()
endfunction()

function(ryoanji_add_executable_test target)
    cmake_parse_arguments(ARG "MPI" "RANKS" "SOURCES;LINK;COMPILE_OPTIONS;INCLUDE;LABELS" ${ARGN})

    if(NOT ARG_SOURCES)
        message(FATAL_ERROR "ryoanji_add_executable_test: SOURCES required")
    endif()

    add_executable(${target} ${ARG_SOURCES})

    if(ARG_INCLUDE)
        target_include_directories(${target} PRIVATE ${ARG_INCLUDE})
    endif()

    if(DEFINED RYOANJI_TEST_INCLUDE_DIRS)
        target_include_directories(${target} PRIVATE ${RYOANJI_TEST_INCLUDE_DIRS})
    endif()

    if(ARG_LINK)
        target_link_libraries(${target} PRIVATE ${ARG_LINK})
    endif()

    if(ARG_COMPILE_OPTIONS)
        target_compile_options(${target} PRIVATE ${ARG_COMPILE_OPTIONS})
    endif()

    if(ARG_MPI)
        if(NOT ARG_RANKS)
            set(ARG_RANKS 2)
        endif()
        add_test(NAME ${target}
                 COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${ARG_RANKS}
                         ${MPIEXEC_PREFLAGS} $<TARGET_FILE:${target}>)
    else()
        add_test(NAME ${target} COMMAND ${target})
    endif()

    if(ARG_LABELS)
        set_tests_properties(${target} PROPERTIES LABELS "${ARG_LABELS}")
    endif()

    install(TARGETS ${target} RUNTIME DESTINATION ${CMAKE_INSTALL_SBINDIR}/ryoanji)
endfunction()
