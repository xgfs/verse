TARGETS = verse verse-neigh verse-simrank

CXXFLAGS = -std=c++11 -march=native

ifeq ($(CXX), g++)
	CXXFLAGS += -fopenmp -Ofast
else ifeq ($(CXX), icpc)
	CXXFLAGS += -qopenmp -O3 -no-prec-div -ansi-alias -ip -static-intel
else ifeq ($(CXX), clang++)
	CXXFLAGS += -fopenmp=libomp -O3 -Wno-shift-op-parentheses
else
	# use g++ by default
	CXX = g++
	CXXFLAGS += -fopenmp -Ofast
endif

all: $(TARGETS)

$(TARGETS):
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $@.cpp

clean:
	rm -rfv $(TARGETS)

.PHONY: all clean