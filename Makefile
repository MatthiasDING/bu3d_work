DIR_BIN = ./bin

SRC = $(wildcard *.cpp)

OBJ = $(patsubst %.cpp, %.o, ${SRC})

TARGET = demo

BIN_TARGET = ${DIR_BIN}/${TARGET}

CC = g++
CFLAGS = -O2 -fopenmp
LIBS = -L ../../DHX -ldhx -L /usr/local/lib/OpenMesh -lOpenMeshCore -lOpenMeshTools  -lopencv_highgui -lopencv_core 

${BIN_TARGET}:${OBJ} 
	$(CC) $(CFLAGS) -o $@ $(OBJ) $(LIBS) 

${OBJ}:%.o:%.cpp
	$(CC) $(CFLAGS) -c  $< -o $@


.PHONY:clean
clean:
	rm -f $(OBJ)  $(BIN_TARGET)

