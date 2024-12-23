#Testing macro
set(CMAKE_CUDA_ARCHITECTURES 70 75) 

macro (add_test_executable_with_lib _name libs)
	# 查找源文件
	file(GLOB CUSOURCES "${_name}*.cu")
	file(GLOB SOURCES "${_name}.cpp")
	file(GLOB CUDA_SOURCES "${CMAKE_SOURCE_DIR}/mpc_cuda/*.cu")
	add_executable(test_${_name} ${SOURCES} ${CUDA_SOURCES} ${CUSOURCES})
	target_link_libraries(test_${_name} ${EMP-OT_LIBRARIES})
endmacro()

macro (add_test_case _name)
	add_test_executable_with_lib(${_name} "")
  	add_test(NAME ${_name} COMMAND "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/test_${_name}" WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/")
endmacro()

macro (add_test_case_with_run _name)
	add_test_executable_with_lib(${_name} "")
	add_test(NAME ${_name} COMMAND "./run" "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/test_${_name}" WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/")

endmacro()

# Test cases
add_test_case_with_run(test_communication) 
add_test_case_with_run(test_invert)
add_test_case_with_run(test_sort)
add_test_case_with_run(test_fss)
add_test_case_with_run(test_lut)
add_test_case_with_run(test_exp)
add_test_case_with_run(test_topk)
