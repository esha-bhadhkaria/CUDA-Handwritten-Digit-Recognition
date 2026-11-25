NVCC = nvcc
CXXFLAGS = -std=c++17
INCLUDE = -Iinclude

TARGET = digit_recognition
SOURCES = src/main.cpp src/neural_network.cu src/data_loader.cpp

all:
	$(NVCC) $(CXXFLAGS) $(INCLUDE) $(SOURCES) -o $(TARGET)

run: all
	./$(TARGET)

clean:
	rm -f $(TARGET)
