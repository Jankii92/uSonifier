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
include CMakeFiles/Scene.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/Scene.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/Scene.dir/flags.make

CMakeFiles/Scene.dir/src/Scene/Scene.cpp.o: CMakeFiles/Scene.dir/flags.make
CMakeFiles/Scene.dir/src/Scene/Scene.cpp.o: ../src/Scene/Scene.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/ubuntu/Documents/Git/uSonifier/bin/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/Scene.dir/src/Scene/Scene.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/Scene.dir/src/Scene/Scene.cpp.o -c /home/ubuntu/Documents/Git/uSonifier/src/Scene/Scene.cpp

CMakeFiles/Scene.dir/src/Scene/Scene.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Scene.dir/src/Scene/Scene.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/ubuntu/Documents/Git/uSonifier/src/Scene/Scene.cpp > CMakeFiles/Scene.dir/src/Scene/Scene.cpp.i

CMakeFiles/Scene.dir/src/Scene/Scene.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Scene.dir/src/Scene/Scene.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/ubuntu/Documents/Git/uSonifier/src/Scene/Scene.cpp -o CMakeFiles/Scene.dir/src/Scene/Scene.cpp.s

CMakeFiles/Scene.dir/src/Scene/Scene.cpp.o.requires:
.PHONY : CMakeFiles/Scene.dir/src/Scene/Scene.cpp.o.requires

CMakeFiles/Scene.dir/src/Scene/Scene.cpp.o.provides: CMakeFiles/Scene.dir/src/Scene/Scene.cpp.o.requires
	$(MAKE) -f CMakeFiles/Scene.dir/build.make CMakeFiles/Scene.dir/src/Scene/Scene.cpp.o.provides.build
.PHONY : CMakeFiles/Scene.dir/src/Scene/Scene.cpp.o.provides

CMakeFiles/Scene.dir/src/Scene/Scene.cpp.o.provides.build: CMakeFiles/Scene.dir/src/Scene/Scene.cpp.o

# Object files for target Scene
Scene_OBJECTS = \
"CMakeFiles/Scene.dir/src/Scene/Scene.cpp.o"

# External object files for target Scene
Scene_EXTERNAL_OBJECTS =

libScene.a: CMakeFiles/Scene.dir/src/Scene/Scene.cpp.o
libScene.a: CMakeFiles/Scene.dir/build.make
libScene.a: CMakeFiles/Scene.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX static library libScene.a"
	$(CMAKE_COMMAND) -P CMakeFiles/Scene.dir/cmake_clean_target.cmake
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/Scene.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/Scene.dir/build: libScene.a
.PHONY : CMakeFiles/Scene.dir/build

CMakeFiles/Scene.dir/requires: CMakeFiles/Scene.dir/src/Scene/Scene.cpp.o.requires
.PHONY : CMakeFiles/Scene.dir/requires

CMakeFiles/Scene.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/Scene.dir/cmake_clean.cmake
.PHONY : CMakeFiles/Scene.dir/clean

CMakeFiles/Scene.dir/depend:
	cd /home/ubuntu/Documents/Git/uSonifier/bin && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ubuntu/Documents/Git/uSonifier /home/ubuntu/Documents/Git/uSonifier /home/ubuntu/Documents/Git/uSonifier/bin /home/ubuntu/Documents/Git/uSonifier/bin /home/ubuntu/Documents/Git/uSonifier/bin/CMakeFiles/Scene.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/Scene.dir/depend

