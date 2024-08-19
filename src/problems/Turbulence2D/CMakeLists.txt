if (AMReX_SPACEDIM EQUAL 3)
    add_executable(test_turbulence2D test_turbulence.cpp ${QuokkaObjSources}
            test_turbulence.cpp
            test_turbulence.cpp)
    if(AMReX_GPU_BACKEND MATCHES "CUDA")
        setup_target_for_cuda_compilation(turbulence2D)
    endif()

    add_test(NAME turbulence2D COMMAND test_turbulence2D test_turbulence2D.in ${QuokkaTestParams} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
endif()