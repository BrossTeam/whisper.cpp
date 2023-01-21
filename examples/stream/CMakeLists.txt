    
set(CMAKE_PREFIX_PATH "C:/dev/vcpkg/installed/x64-windows/share")
# stream
set(TARGET stream)
find_package(SDL2 CONFIG REQUIRED)
add_executable(${TARGET} stream.cpp)

include(DefaultTargetOptions)

target_link_libraries(${TARGET} PRIVATE whisper ${CMAKE_THREAD_LIBS_INIT})

target_link_libraries(${TARGET} 
    PRIVATE
    $<TARGET_NAME_IF_EXISTS:SDL2::SDL2main>
    $<IF:$<TARGET_EXISTS:SDL2::SDL2>,SDL2::SDL2,SDL2::SDL2-static>
)


