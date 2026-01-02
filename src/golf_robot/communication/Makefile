CXX = g++
CXXFLAGS = -std=c++17 -O2 -pthread -Wall -Wextra -Iinclude -Iinclude/communication
LDFLAGS = -lgsl -lgslcblas -lm -pthread

# Sources live in src/
COMMON_SRCS = \
    src/ur_driver.cpp \
    src/ur_communication.cpp \
    src/ur_realtime_communication.cpp \
    src/robot_state.cpp \
    src/robot_state_RT.cpp \
    src/do_output.cpp \
    src/ufwdkin_c.c \
    src/ujac_c.c \
# 	src/kf_tuning.cpp

# TRAJ_SRCS  = src/traj_streamer.cpp
TRAJ_SRCS  = src/traj_streamer_swing.cpp
TRAJ2_SRCS = src/traj_streamer_2.cpp
TEST_SRCS  = src/test.cpp

COMMON_OBJS = $(COMMON_SRCS:.cpp=.o)
COMMON_OBJS := $(COMMON_OBJS:.c=.o)
TRAJ_OBJS   = $(TRAJ_SRCS:.cpp=.o)
TRAJ2_OBJS  = $(TRAJ2_SRCS:.cpp=.o)
TEST_OBJS   = $(TEST_SRCS:.cpp=.o)

DRY_TARGET  = traj_streamer_dry
# TRAJ_TARGET = traj_streamer
TRAJ_TARGET = traj_streamer_swing
TRAJ2_TARGET = traj_streamer2
TEST_TARGET = test_runner

.PHONY: all traj dry-build traj2 test build clean run-dry

all: traj

# Dry build: compile the common sources + main streamer implementation
dry-build: CXXFLAGS += -DNO_URDRIVER
dry-build: $(DRY_TARGET)

$(DRY_TARGET): $(COMMON_OBJS) $(TRAJ_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

traj: $(TRAJ_TARGET)

$(TRAJ_TARGET): $(COMMON_OBJS) $(TRAJ_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

traj2: $(TRAJ2_TARGET)

$(TRAJ2_TARGET): $(COMMON_OBJS) $(TRAJ2_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

test: $(TEST_TARGET)

$(TEST_TARGET): $(COMMON_OBJS) $(TEST_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

# Pattern rules – work with paths like src/foo.cpp → src/foo.o
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.o: %.c
	# force C language when compiling .c files so symbols have C linkage (no C++ name-mangling)
	$(CXX) -x c $(CXXFLAGS) -c $< -o $@

run-dry: dry-build
	@echo "Running dry streamer against log/trajectory_sim.csv (if present)"
	./$(DRY_TARGET) log/trajectory_sim.csv 127.0.0.1 1

clean:
	rm -f src/*.o $(DRY_TARGET) $(TRAJ_TARGET) $(TRAJ2_TARGET) $(TEST_TARGET)
