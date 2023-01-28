    
set(CMAKE_PREFIX_PATH "C:/dev/vcpkg/installed/x64-windows/share")
# stream
set(TARGET stream)
find_package(SDL2 CONFIG REQUIRED)

#set(Boost_INCLUDE_DIR "C:/dev/vcpkg/installed/x64-windows/include/boost")  
#find_package(Boost CONFIG REQUIRED COMPONENTS circular_buffer)
add_executable(${TARGET} "stream.cpp" "audio.h")

include(DefaultTargetOptions)

target_link_libraries(${TARGET} PRIVATE whisper ${CMAKE_THREAD_LIBS_INIT})

target_link_libraries(${TARGET} 
    PRIVATE
    $<TARGET_NAME_IF_EXISTS:SDL2::SDL2main>
    $<IF:$<TARGET_EXISTS:SDL2::SDL2>,SDL2::SDL2,SDL2::SDL2-static>
)


