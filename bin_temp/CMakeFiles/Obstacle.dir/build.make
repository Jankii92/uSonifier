# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The program to use to edit the cache.
CMAKE_EDIT_COMMAND = /usr/bin/ccmake

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/ubuntu/Documents/Git/uSonifier

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/ubuntu/Documents/Git/uSonifier/bin

# Include any dependencies generated for this target.
include CMakeFiles/Obstacle.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/Obstacle.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/Obstacle.dir/flags.make

CMakeFiles/Obstacle.dir/src/Scene/Obstacle.cpp.o: CMakeFiles/Obstacle.dir/flags.make
CMakeFiles/Obstacle.dir/src/Scene/Obstacle.cpp.o: ../src/Scene/Obstacle.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/ubuntu/Documents/Git/uSonifier/bin/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/Obstacle.dir/src/Scene/Obstacle.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/Obstacle.dir/src/Scene/Obstacle.cpp.o -c /home/ubuntu/Documents/Git/uSonifier/src/Scene/Obstacle.cpp

CMakeFiles/Obstacle.dir/src/Scene/Obstacle.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Obstacle.dir/src/Scene/Obstacle.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/ubuntu/Documents/Git/uSonifier/src/Scene/Obstacle.cpp > CMakeFiles/Obstacle.dir/src/Scene/Obstacle.cpp.i

CMakeFiles/Obstacle.dir/src/Scene/Obstacle.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Obstacle.dir/src/Scene/Obstacle.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/ubuntu/Documents/Git/uSonifier/src/Scene/Obstacle.cpp -o CMakeFiles/Obstacle.dir/src/Scene/Obstacle.cpp.s

CMakeFiles/Obstacle.dir/src/Scene/Obstacle.cpp.o.requires:
.PHONY : CMakeFiles/Obstacle.dir/src/Scene/Obstacle.cpp.o.requires

CMakeFiles/Obstacle.dir/src/Scene/Obstacle.cpp.o.provides: CMakeFiles/Obstacle.dir/src/Scene/Obstacle.cpp.o.requires
	$(MAKE) -f CMakeFiles/Obstacle.dir/build.make CMakeFiles/Obstacle.dir/src/Scene/Obstacle.cpp.o.provides.build
.PHONY : CMakeFiles/Obstacle.dir/src/Scene/Obstacle.cpp.o.provides

CMakeFiles/Obstacle.dir/src/Scene/Obstacle.cpp.o.provides.build: CMakeFiles/Obstacle.dir/src/Scene/Obstacle.cpp.o

# Object files for target Obstacle
Obstacle_OBJECTS = \
"CMakeFiles/Obstacle.dir/src/Scene/Obstacle.cpp.o"

# External object files for target Obstacle
Obstacle_EXTERNAL_OBJECTS =

libObstacle.a: CMakeFiles/Obstacle.dir/src/Scene/Obstacle.cpp.o
libObstacle.a: CMakeFiles/Obstacle.dir/build.make
libObstacle.a: CMakeFiles/Obstacle.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX static library libObstacle.a"
	$(CMAKE_COMMAND) -P CMakeFiles/Obstacle.dir/cmake_clean_target.cmake
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/Obstacle.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/Obstacle.dir/build: libObstacle.a
.PHONY : CMakeFiles/Obstacle.dir/build

CMakeFiles/Obstacle.dir/requires: CMakeFiles/Obstacle.dir/src/Scene/Obstacle.cpp.o.requires
.PHONY : CMakeFiles/Obstacle.dir/requires

CMakeFiles/Obstacle.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/Obstacle.dir/cmake_clean.cmake
.PHONY : CMakeFiles/Obstacle.dir/clean

CMakeFiles/Obstacle.dir/depend:
	cd /home/ubuntu/Documents/Git/uSonifier/bin && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ubuntu/Documents/Git/uSonifier /home/ubuntu/Documents/Git/uSonifier /home/ubuntu/Documents/Git/uSonifier/bin /home/ubuntu/Documents/Git/uSonifier/bin /home/ubuntu/Documents/Git/uSonifier/bin/CMakeFiles/Obstacle.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/Obstacle.dir/depend

