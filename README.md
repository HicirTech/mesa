# PanVK 9300

This branch, `panvk-9300`, is upstream Mesa `main` plus a kbase backend for
PanVK: Mesa's Vulkan driver for Arm Mali runs on stock Android devices
through `/dev/mali0`, without root. The target is the MediaTek Dimensity
9300+ (Immortalis-G720 MC12) running Windows games through Proton, DXVK and
vkd3d-proton. Other Mali GPUs that the imported backend supports should keep
working but are not tested on this branch.

Not a conformant Vulkan implementation; for testing.

## Where the code comes from

- Upstream Mesa `main` (https://gitlab.freedesktop.org/mesa/mesa).
- The kbase backend (job manager and CSF), the PanVK queue and sync changes,
  gralloc/AHardwareBuffer import, BCn emulation, geometry, tessellation and
  transform feedback on compute, and the Android build and packaging scripts
  come from [mali-panvk](https://github.com/JimVulkan/mali-panvk) by
  JimVulkan at `58f2c55`, imported in one commit and rebased onto `main`.
  Its own documentation is in
  [android/README-mali-panvk.md](android/README-mali-panvk.md).
- This branch's own changes are the commits after that import.

## Building

On Linux or WSL:

```sh
./build.sh /path/to/android-ndk
```

Requirements are those listed in `android/README-mali-panvk.md` (Android NDK
r29 tested, meson, ninja, Python with mako, packaging and pyyaml,
glslangValidator, and LLVM with clang, libclc and the SPIR-V LLVM
translator for the host tools), plus `rustc` with the
`aarch64-linux-android` target and `bindgen` for the Kraid compiler.
`KRAID=0` builds without Kraid and without Rust.

The result is `dist/panvk-9300-<version>.zip` with `meta.json`,
`libvulkan_panfrost.so` and `NOTICE.txt`, the package format of
adrenotools-style driver loaders.

## Runtime switches

| Environment | Effect |
|---|---|
| `PAN_USE_KRAID=all` (or `cs`, `fs`, `vs`, `internal`) | compiles with Kraid instead of the C compiler; off by default, see `src/panfrost/compiler/pan_compiler.c` |
| `PANVK_DEBUG=startup` | logs why the device could not be created |
| `PANVK_FORCE_IDVS=1` | turns index-driven vertex shading back on (off by default on Bifrost) |
| `PANVK_NO_MEDIUMP=1` | compiles `mediump` shader code at 32 bits |
| `MESA_SHADER_CACHE_DISABLE=true` | turns the disk shader cache off |
| `MESA_LOG=file` | writes the driver's log, including GPU fault details, to a file |

## License

Mesa files keep the licenses in their headers (see `docs/license.rst`). The
changes imported from mali-panvk are MIT (`LICENSE`); the AOSP gralloc
headers are Apache-2.0 (`LICENSES/`, `THIRD_PARTY_NOTICES.md`). Driver
packages carry `android/NOTICE.txt`, which covers everything compiled into
`libvulkan_panfrost.so`.
