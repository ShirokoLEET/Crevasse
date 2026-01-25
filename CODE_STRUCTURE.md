# Crevasse - Code Structure Documentation

## Overview

Crevasse is an OBS Studio plugin built using the OBS Plugin Template. This document describes the architecture and organization of the codebase.

## Directory Structure

```
Crevasse/
├── .github/                 # GitHub-specific files
│   ├── actions/            # Custom GitHub Actions
│   ├── scripts/            # Build and CI scripts
│   └── workflows/          # CI/CD workflow definitions
├── build-aux/              # Build auxiliary files
├── cmake/                  # CMake build system configuration
│   ├── common/            # Common CMake modules
│   ├── linux/             # Linux-specific configuration
│   ├── macos/             # macOS-specific configuration
│   └── windows/           # Windows-specific configuration
├── data/                   # Plugin data files
│   └── locale/            # Localization/translation files
├── src/                    # Source code
│   ├── plugin-main.c      # Main plugin entry point
│   ├── plugin-support.c.in # Plugin support functions (template)
│   └── plugin-support.h   # Plugin support header
├── buildspec.json          # Plugin build specification
├── CMakeLists.txt         # Main CMake configuration
├── CMakePresets.json      # CMake build presets
└── README.md              # Project documentation

```

## Core Components

### 1. Build Configuration

#### buildspec.json
The main plugin metadata file containing:
- **name**: Internal plugin identifier (`crevasse`)
- **displayName**: User-facing plugin name (`Crevasse`)
- **version**: Plugin version following semantic versioning
- **author**: Plugin author (`@ShirokoLEET`)
- **website**: Project website/repository
- **email**: Contact email
- **dependencies**: Build dependencies (OBS Studio, prebuilt deps, Qt6)
- **platformConfig**: Platform-specific settings (e.g., macOS bundle ID)

#### CMakeLists.txt
Main CMake build configuration that:
- Defines the project and version (read from buildspec.json)
- Sets up build options (ENABLE_FRONTEND_API, ENABLE_QT)
- Configures compiler settings
- Links required libraries (libobs, obs-frontend-api, Qt6)
- Defines source files and build targets

### 2. Source Code

#### plugin-main.c
The main entry point for the OBS plugin:
- **OBS_DECLARE_MODULE()**: Declares this as an OBS module
- **OBS_MODULE_USE_DEFAULT_LOCALE()**: Sets up localization
- **obs_module_load()**: Called when the plugin is loaded
- **obs_module_unload()**: Called when the plugin is unloaded

#### plugin-support.h
Header file defining:
- Plugin constants (PLUGIN_NAME, PLUGIN_VERSION)
- Logging function declarations
- Cross-platform compatibility macros

#### plugin-support.c.in
CMake template file that:
- Defines PLUGIN_NAME and PLUGIN_VERSION constants
- Implements the obs_log() wrapper function
- Provides formatted logging with plugin name prefix

### 3. CMake Build System

#### cmake/common/
Common CMake modules shared across all platforms:

- **bootstrap.cmake**: 
  - Initializes the build system
  - Reads buildspec.json and extracts metadata
  - Sets up project variables (author, website, version)
  - Configures build types

- **buildspec_common.cmake**:
  - Manages external dependencies
  - Downloads and sets up OBS Studio sources
  - Configures prebuilt dependencies and Qt6
  - Handles dependency versioning and caching

- **helpers_common.cmake**:
  - Provides utility functions for the build system

- **compiler_common.cmake**:
  - Sets compiler flags and warnings
  - Configures platform-specific compiler settings

- **osconfig.cmake**:
  - Detects and configures OS-specific settings

- **buildnumber.cmake**:
  - Manages build numbers and versioning

- **ccache.cmake**:
  - Configures ccache for faster rebuilds

#### cmake/{linux,macos,windows}/
Platform-specific CMake configurations for each supported operating system.

### 4. GitHub Actions & CI/CD

#### .github/workflows/
Automated workflows for continuous integration:

- **push.yaml**: Runs on commits to main/master branches
- **pr-pull.yaml**: Runs on pull request updates
- **dispatch.yaml**: Manual workflow trigger
- **build-project.yaml**: Builds the plugin across platforms
- **check-format.yaml**: Validates code formatting

#### .github/actions/
Custom reusable GitHub Actions for build tasks

#### .github/scripts/
Build scripts used by GitHub Actions workflows

### 5. Data Files

#### data/locale/
Contains translation files for internationalization (i18n):
- Organized by locale codes (e.g., en-US, de-DE)
- INI format for translation strings
- Loaded via OBS_MODULE_USE_DEFAULT_LOCALE macro

## Build Process

1. **Configuration Phase**:
   - CMake reads buildspec.json
   - Extracts plugin metadata (name, version, author)
   - Downloads required dependencies if needed
   - Configures platform-specific settings

2. **Generation Phase**:
   - Processes plugin-support.c.in → plugin-support.c
   - Substitutes @CMAKE_PROJECT_NAME@ and @CMAKE_PROJECT_VERSION@
   - Generates platform-specific build files

3. **Build Phase**:
   - Compiles source files
   - Links against libobs and other dependencies
   - Creates plugin module/library

4. **Package Phase** (for releases):
   - Creates platform-specific installers
   - Generates distributable packages

## Plugin Lifecycle

1. **Loading**:
   ```c
   obs_module_load() → Initialize plugin → Return true on success
   ```

2. **Running**:
   - Plugin registers sources, filters, outputs, etc. (to be implemented)
   - Responds to OBS events

3. **Unloading**:
   ```c
   obs_module_unload() → Cleanup resources → Plugin removed
   ```

## Adding New Features

### To add a new source/filter:
1. Create implementation file in `src/`
2. Add to `target_sources()` in CMakeLists.txt
3. Register in `obs_module_load()`
4. Add localization strings to `data/locale/`

### To add Qt UI:
1. Set `ENABLE_QT=ON` in CMake
2. Create .ui files or Qt widgets
3. Link against Qt6 (already configured)
4. Use obs-frontend-api for OBS UI integration

### To add dependencies:
1. Update buildspec.json with dependency info
2. Modify cmake/common/buildspec_common.cmake if needed
3. Add find_package() and target_link_libraries() to CMakeLists.txt

## Key Features

- **Cross-platform**: Supports Windows, macOS, and Linux
- **CMake-based**: Modern build system with presets
- **CI/CD ready**: GitHub Actions for automated testing and releases
- **Localization support**: Built-in i18n framework
- **Modular**: Easy to extend with new sources/filters
- **Template-based**: Uses CMake configure_file for metadata injection

## Development Tools

- **CMake**: Build system generator
- **clang-format**: Code formatting (see .clang-format)
- **gersemi**: CMake formatting (see .gersemirc)
- **GitHub Actions**: Automated CI/CD

## Plugin Metadata Flow

```
buildspec.json
    ↓
bootstrap.cmake (reads JSON, sets variables)
    ↓
CMakeLists.txt (uses variables in project())
    ↓
plugin-support.c.in (configured with CMake variables)
    ↓
plugin-support.c (generated with actual values)
    ↓
plugin-main.c (uses PLUGIN_NAME constant)
```

## Support & Resources

- **Repository**: https://github.com/ShirokoLEET/Crevasse
- **OBS Studio**: https://obsproject.com
- **Plugin Template Wiki**: https://github.com/obsproject/obs-plugintemplate/wiki

---

*Document maintained by @ShirokoLEET*
