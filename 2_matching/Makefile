# 2024-09-19  Ian M. Hay-Arthur
#
# Compile and run the main program.

# Outputs in $(BINDIR)/ and $(OJ)/

SHELL      = /bin/sh
BINDIR     = ./bin
OJ         = ./obj
JSON_LIB   = ./libs/json/include

CFLAGS     := $(shell pkg-config --cflags opencv4)
LIBS       := $(shell pkg-config --libs opencv4)

CXXFLAGS   = -g -Wall -std=c++17 -I$(JSON_LIB)

%: %.cpp
	g++ $(CFLAGS) $(LIBS) -o $@ $<

# compile .o files without linking
obj: $(OJ)/main.o

run: compile
	# $(BINDIR)/snowMatching --TESTOP

compile: submodule $(OJ) $(BINDIR) $(BINDIR)/snowMatching

mkdirs: $(OJ) $(BINDIR)

clean:
	rm -f $(OJ)/*

clobber:
	rm -f $(OJ)/*
	rm -f $(BINDIR)/snowMatching

# Ensure submodule is initialized and present
submodule:
	mkdir -p libs
	git submodule update --init --recursive

$(BINDIR):
	mkdir -p $(BINDIR)

$(OJ):
	mkdir -p $(OJ)

$(BINDIR)/snowMatching: $(OJ)/main.o $(OJ)/utils.o $(OJ)/matchingCode.o $(OJ)/fundamentalMatrix.o
	g++ -o $@ $(OJ)/*.o $(LIBS) $(CFLAGS) -Wl,-rpath=/home/nvidia/SnowTeam/opencv/build/lib

$(OJ)/main.o: src/main.cc $(OJ)/utils.o $(OJ)/matchingCode.o $(OJ)/fundamentalMatrix.o
	g++ $(CXXFLAGS) $(CFLAGS) -c src/main.cc -o $@

$(OJ)/matchingCode.o: src/matchingCode.cc src/matchingCode.h $(OJ)/utils.o
	g++ $(CXXFLAGS) $(CFLAGS) -c src/matchingCode.cc -o $@

$(OJ)/fundamentalMatrix.o: src/fundamentalMatrix.cc src/fundamentalMatrix.h $(OJ)/utils.o
	g++ $(CXXFLAGS) $(CFLAGS) -c src/fundamentalMatrix.cc -o $@

$(OJ)/utils.o: src/utils.cc src/utils.h
	g++ $(CXXFLAGS) $(CFLAGS) -c src/utils.cc -o $@