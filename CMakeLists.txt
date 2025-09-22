cmake_minimum_required(VERSION 3.10)
project(RhoCore_solver LANGUAGES CXX CUDA)

# Pfad zur nlohmann/json.hpp-Bibliothek
# Annahme: json.hpp befindet sich im selben Verzeichnis wie CMakeLists.txt
set(NLOHMANN_JSON_PATH "${CMAKE_SOURCE_DIR}")

# Konfiguriert die Kompilierungs-Flags
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES "120") # Optimiert für Ihre RTX 5090

# Fügt die Header-Pfade hinzu
include_directories(${NLOHMANN_JSON_PATH})

# Definiert das Haupt-Executable
add_executable(RhoCore_solver RhoCore_solver.cu)

# Linkt die CUDA-Laufzeitbibliothek
target_link_libraries(RhoCore_solver PRIVATE CUDA::cudart)

# Erstellt ein benutzerdefiniertes Ziel, um die Datenbank vorzubereiten
add_custom_target(
    prepare_db
    COMMAND python3 bitcoin_key_search.py
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Downloading and sorting Bitcoin addresses..."
)

# Stellt sicher, dass das prepare_db-Ziel vor dem Hauptprogramm ausgeführt wird
add_dependencies(RhoCore_solver prepare_db)

# Optional: Installiert das Executable, um es einfach ausführen zu können
install(TARGETS RhoCore_solver DESTINATION bin)
