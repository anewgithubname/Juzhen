#pragma section("__nv_managed_data__")
static char __nv_inited_managed_rt = 0; static void **__nv_fatbinhandle_for_managed_rt; static void __nv_save_fatbinhandle_for_managed_rt(void **in){__nv_fatbinhandle_for_managed_rt = in;} static char __nv_init_managed_rt_with_module(void **); static inline void __nv_init_managed_rt(void) { __nv_inited_managed_rt = (__nv_inited_managed_rt ? __nv_inited_managed_rt                 : __nv_init_managed_rt_with_module(__nv_fatbinhandle_for_managed_rt));}
#line 1 "CMakeCUDACompilerId.cu"
#define __nv_is_extended_device_lambda_closure_type(X) false
#define __nv_is_extended_host_device_lambda_closure_type(X) false
#if defined(__nv_is_extended_device_lambda_closure_type) && defined(__nv_is_extended_host_device_lambda_closure_type)
#endif

#line 1
#line 67 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/bin/../include\\cuda_runtime.h"
#pragma warning(push)
#pragma warning(disable: 4820)
#line 708 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Enterprise\\VC\\Tools\\MSVC\\14.37.32822\\include\\sal.h"
#pragma region Input Buffer SAL 1 compatibility macros
#line 1472
#pragma endregion Input Buffer SAL 1 compatibility macros
#line 2361 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Enterprise\\VC\\Tools\\MSVC\\14.37.32822\\include\\sal.h"
extern "C" {
#line 2971 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Enterprise\\VC\\Tools\\MSVC\\14.37.32822\\include\\sal.h"
}
#line 22 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Enterprise\\VC\\Tools\\MSVC\\14.37.32822\\include\\concurrencysal.h"
extern "C" {
#line 391 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Enterprise\\VC\\Tools\\MSVC\\14.37.32822\\include\\concurrencysal.h"
}
#line 15 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Enterprise\\VC\\Tools\\MSVC\\14.37.32822\\include\\vadefs.h"
#pragma pack ( push, 8 )
#line 47 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Enterprise\\VC\\Tools\\MSVC\\14.37.32822\\include\\vadefs.h"
#pragma warning(push)
#pragma warning(disable: 4514 4820 )
#line 51
extern "C" {
#line 61 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Enterprise\\VC\\Tools\\MSVC\\14.37.32822\\include\\vadefs.h"
typedef unsigned __int64 uintptr_t; 
#line 72 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Enterprise\\VC\\Tools\\MSVC\\14.37.32822\\include\\vadefs.h"
typedef char *va_list; 
#line 155 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Enterprise\\VC\\Tools\\MSVC\\14.37.32822\\include\\vadefs.h"
void __cdecl __va_start(va_list *, ...); 
#line 167 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Enterprise\\VC\\Tools\\MSVC\\14.37.32822\\include\\vadefs.h"
}
#line 171 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Enterprise\\VC\\Tools\\MSVC\\14.37.32822\\include\\vadefs.h"
extern "C++" {
#line 173
template< class _Ty> 
#line 174
struct __vcrt_va_list_is_reference { 
#line 176
enum: bool { __the_value}; 
#line 177
}; 
#line 179
template< class _Ty> 
#line 180
struct __vcrt_va_list_is_reference< _Ty &>  { 
#line 182
enum: bool { __the_value = '\001'}; 
#line 183
}; 
#line 185
template< class _Ty> 
#line 186
struct __vcrt_va_list_is_reference< _Ty &&>  { 
#line 188
enum: bool { __the_value = '\001'}; 
#line 189
}; 
#line 191
template< class _Ty> 
#line 192
struct __vcrt_assert_va_start_is_not_reference { 
#line 194
static_assert((!__vcrt_va_list_is_reference< _Ty> ::__the_value), "va_start argument must not have reference type and must not be parenthesized");
#line 196
}; 
#line 197
}
#line 207 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Enterprise\\VC\\Tools\\MSVC\\14.37.32822\\include\\vadefs.h"
#pragma warning(pop)
#pragma pack ( pop )
#line 60 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Enterprise\\VC\\Tools\\MSVC\\14.37.32822\\include\\vcruntime.h"
#pragma warning(push)
#pragma warning(disable: 4514 4820 )
#line 96 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Enterprise\\VC\\Tools\\MSVC\\14.37.32822\\include\\vcruntime.h"
__pragma( pack ( push, 8 )) extern "C" {
#line 193 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Enterprise\\VC\\Tools\\MSVC\\14.37.32822\\include\\vcruntime.h"
typedef unsigned __int64 size_t; 
#if !defined(__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__)
#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#endif
#include "crt/host_runtime.h"
#line 194
typedef __int64 ptrdiff_t; 
#line 195
typedef __int64 intptr_t; 
#line 203 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Enterprise\\VC\\Tools\\MSVC\\14.37.32822\\include\\vcruntime.h"
typedef bool __vcrt_bool; 
#line 250 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Enterprise\\VC\\Tools\\MSVC\\14.37.32822\\include\\vcruntime.h"
extern "C++" {
#line 252
template< class _CountofType, size_t _SizeOfArray> char (*__countof_helper(__unaligned _CountofType (& _Array)[_SizeOfArray]))[_SizeOfArray]; 
#line 256
}
#line 377 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Enterprise\\VC\\Tools\\MSVC\\14.37.32822\\include\\vcruntime.h"
void __cdecl __security_init_cookie(); 
#line 386 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Enterprise\\VC\\Tools\\MSVC\\14.37.32822\\include\\vcruntime.h"
void __cdecl __security_check_cookie(uintptr_t _StackCookie); 
#line 387
__declspec(noreturn) void __cdecl __report_gsfailure(uintptr_t _StackCookie); 
#line 391 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Enterprise\\VC\\Tools\\MSVC\\14.37.32822\\include\\vcruntime.h"
extern uintptr_t __security_cookie; 
#line 399 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Enterprise\\VC\\Tools\\MSVC\\14.37.32822\\include\\vcruntime.h"
}__pragma( pack ( pop )) 
#line 401
#pragma warning(pop)
#line 73 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\corecrt.h"
#pragma warning(push)
#pragma warning(disable: 4324 4514 4574 4710 4793 4820 4995 4996 28719 28726 28727 )
#line 77
__pragma( pack ( push, 8 )) extern "C" {
#line 199 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\corecrt.h"
extern "C++" {
#line 201
template< bool _Enable, class _Ty> struct _CrtEnableIf; 
#line 204
template< class _Ty> 
#line 205
struct _CrtEnableIf< true, _Ty>  { 
#line 207
typedef _Ty _Type; 
#line 208
}; 
#line 209
}
#line 213 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\corecrt.h"
typedef bool __crt_bool; 
#line 357 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\corecrt.h"
void __cdecl _invalid_parameter_noinfo(); 
#line 358
__declspec(noreturn) void __cdecl _invalid_parameter_noinfo_noreturn(); 
#line 360
__declspec(noreturn) void __cdecl 
#line 361
_invoke_watson(const __wchar_t * _Expression, const __wchar_t * _FunctionName, const __wchar_t * _FileName, unsigned _LineNo, uintptr_t _Reserved); 
#line 590 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\corecrt.h"
typedef int errno_t; 
#line 591
typedef unsigned short wint_t; 
#line 592
typedef unsigned short wctype_t; 
#line 593
typedef long __time32_t; 
#line 594
typedef __int64 __time64_t; 
#line 601
typedef 
#line 596
struct __crt_locale_data_public { 
#line 598
const unsigned short *_locale_pctype; 
#line 599
int _locale_mb_cur_max; 
#line 600
unsigned _locale_lc_codepage; 
#line 601
} __crt_locale_data_public; 
#line 607
typedef 
#line 603
struct __crt_locale_pointers { 
#line 605
struct __crt_locale_data *locinfo; 
#line 606
struct __crt_multibyte_data *mbcinfo; 
#line 607
} __crt_locale_pointers; 
#line 609
typedef __crt_locale_pointers *_locale_t; 
#line 615
typedef 
#line 611
struct _Mbstatet { 
#line 613
unsigned long _Wchar; 
#line 614
unsigned short _Byte, _State; 
#line 615
} _Mbstatet; 
#line 617
typedef _Mbstatet mbstate_t; 
#line 631 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\corecrt.h"
typedef __time64_t time_t; 
#line 641 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\corecrt.h"
typedef size_t rsize_t; 
#line 2058 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\corecrt.h"
}__pragma( pack ( pop )) 
#line 2061
#pragma warning(pop)
#line 68 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\device_types.h"
#if 0
#line 68
enum cudaRoundMode { 
#line 70
cudaRoundNearest, 
#line 71
cudaRoundZero, 
#line 72
cudaRoundPosInf, 
#line 73
cudaRoundMinInf
#line 74
}; 
#endif
#line 100 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\vector_types.h"
#if 0
#line 100
struct char1 { 
#line 102
signed char x; 
#line 103
}; 
#endif
#line 105 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\vector_types.h"
#if 0
#line 105
struct uchar1 { 
#line 107
unsigned char x; 
#line 108
}; 
#endif
#line 111 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\vector_types.h"
#if 0
#line 111
struct __declspec(align(2)) char2 { 
#line 113
signed char x, y; 
#line 114
}; 
#endif
#line 116 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\vector_types.h"
#if 0
#line 116
struct __declspec(align(2)) uchar2 { 
#line 118
unsigned char x, y; 
#line 119
}; 
#endif
#line 121 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\vector_types.h"
#if 0
#line 121
struct char3 { 
#line 123
signed char x, y, z; 
#line 124
}; 
#endif
#line 126 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\vector_types.h"
#if 0
#line 126
struct uchar3 { 
#line 128
unsigned char x, y, z; 
#line 129
}; 
#endif
#line 131 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\vector_types.h"
#if 0
#line 131
struct __declspec(align(4)) char4 { 
#line 133
signed char x, y, z, w; 
#line 134
}; 
#endif
#line 136 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\vector_types.h"
#if 0
#line 136
struct __declspec(align(4)) uchar4 { 
#line 138
unsigned char x, y, z, w; 
#line 139
}; 
#endif
#line 141 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\vector_types.h"
#if 0
#line 141
struct short1 { 
#line 143
short x; 
#line 144
}; 
#endif
#line 146 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\vector_types.h"
#if 0
#line 146
struct ushort1 { 
#line 148
unsigned short x; 
#line 149
}; 
#endif
#line 151 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\vector_types.h"
#if 0
#line 151
struct __declspec(align(4)) short2 { 
#line 153
short x, y; 
#line 154
}; 
#endif
#line 156 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\vector_types.h"
#if 0
#line 156
struct __declspec(align(4)) ushort2 { 
#line 158
unsigned short x, y; 
#line 159
}; 
#endif
#line 161 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\vector_types.h"
#if 0
#line 161
struct short3 { 
#line 163
short x, y, z; 
#line 164
}; 
#endif
#line 166 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\vector_types.h"
#if 0
#line 166
struct ushort3 { 
#line 168
unsigned short x, y, z; 
#line 169
}; 
#endif
#line 171 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\vector_types.h"
#if 0
#line 171
struct __declspec(align(8)) short4 { short x; short y; short z; short w; }; 
#endif
#line 172 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\vector_types.h"
#if 0
#line 172
struct __declspec(align(8)) ushort4 { unsigned short x; unsigned short y; unsigned short z; unsigned short w; }; 
#endif
#line 174 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\vector_types.h"
#if 0
#line 174
struct int1 { 
#line 176
int x; 
#line 177
}; 
#endif
#line 179 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\vector_types.h"
#if 0
#line 179
struct uint1 { 
#line 181
unsigned x; 
#line 182
}; 
#endif
#line 184 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\vector_types.h"
#if 0
#line 184
struct __declspec(align(8)) int2 { int x; int y; }; 
#endif
#line 185 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\vector_types.h"
#if 0
#line 185
struct __declspec(align(8)) uint2 { unsigned x; unsigned y; }; 
#endif
#line 187 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\vector_types.h"
#if 0
#line 187
struct int3 { 
#line 189
int x, y, z; 
#line 190
}; 
#endif
#line 192 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\vector_types.h"
#if 0
#line 192
struct uint3 { 
#line 194
unsigned x, y, z; 
#line 195
}; 
#endif
#line 197 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\vector_types.h"
#if 0
#line 197
struct __declspec(align(16)) int4 { 
#line 199
int x, y, z, w; 
#line 200
}; 
#endif
#line 202 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\vector_types.h"
#if 0
#line 202
struct __declspec(align(16)) uint4 { 
#line 204
unsigned x, y, z, w; 
#line 205
}; 
#endif
#line 207 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\vector_types.h"
#if 0
#line 207
struct long1 { 
#line 209
long x; 
#line 210
}; 
#endif
#line 212 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\vector_types.h"
#if 0
#line 212
struct ulong1 { 
#line 214
unsigned long x; 
#line 215
}; 
#endif
#line 218 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\vector_types.h"
#if 0
#line 218
struct __declspec(align(8)) long2 { long x; long y; }; 
#endif
#line 219 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\vector_types.h"
#if 0
#line 219
struct __declspec(align(8)) ulong2 { unsigned long x; unsigned long y; }; 
#endif
#line 234 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\vector_types.h"
#if 0
#line 234
struct long3 { 
#line 236
long x, y, z; 
#line 237
}; 
#endif
#line 239 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\vector_types.h"
#if 0
#line 239
struct ulong3 { 
#line 241
unsigned long x, y, z; 
#line 242
}; 
#endif
#line 244 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\vector_types.h"
#if 0
#line 244
struct __declspec(align(16)) long4 { 
#line 246
long x, y, z, w; 
#line 247
}; 
#endif
#line 249 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\vector_types.h"
#if 0
#line 249
struct __declspec(align(16)) ulong4 { 
#line 251
unsigned long x, y, z, w; 
#line 252
}; 
#endif
#line 254 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\vector_types.h"
#if 0
#line 254
struct float1 { 
#line 256
float x; 
#line 257
}; 
#endif
#line 276 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\vector_types.h"
#if 0
#line 276
struct __declspec(align(8)) float2 { float x; float y; }; 
#endif
#line 281 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\vector_types.h"
#if 0
#line 281
struct float3 { 
#line 283
float x, y, z; 
#line 284
}; 
#endif
#line 286 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\vector_types.h"
#if 0
#line 286
struct __declspec(align(16)) float4 { 
#line 288
float x, y, z, w; 
#line 289
}; 
#endif
#line 291 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\vector_types.h"
#if 0
#line 291
struct longlong1 { 
#line 293
__int64 x; 
#line 294
}; 
#endif
#line 296 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\vector_types.h"
#if 0
#line 296
struct ulonglong1 { 
#line 298
unsigned __int64 x; 
#line 299
}; 
#endif
#line 301 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\vector_types.h"
#if 0
#line 301
struct __declspec(align(16)) longlong2 { 
#line 303
__int64 x, y; 
#line 304
}; 
#endif
#line 306 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\vector_types.h"
#if 0
#line 306
struct __declspec(align(16)) ulonglong2 { 
#line 308
unsigned __int64 x, y; 
#line 309
}; 
#endif
#line 311 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\vector_types.h"
#if 0
#line 311
struct longlong3 { 
#line 313
__int64 x, y, z; 
#line 314
}; 
#endif
#line 316 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\vector_types.h"
#if 0
#line 316
struct ulonglong3 { 
#line 318
unsigned __int64 x, y, z; 
#line 319
}; 
#endif
#line 321 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\vector_types.h"
#if 0
#line 321
struct __declspec(align(16)) longlong4 { 
#line 323
__int64 x, y, z, w; 
#line 324
}; 
#endif
#line 326 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\vector_types.h"
#if 0
#line 326
struct __declspec(align(16)) ulonglong4 { 
#line 328
unsigned __int64 x, y, z, w; 
#line 329
}; 
#endif
#line 331 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\vector_types.h"
#if 0
#line 331
struct double1 { 
#line 333
double x; 
#line 334
}; 
#endif
#line 336 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\vector_types.h"
#if 0
#line 336
struct __declspec(align(16)) double2 { 
#line 338
double x, y; 
#line 339
}; 
#endif
#line 341 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\vector_types.h"
#if 0
#line 341
struct double3 { 
#line 343
double x, y, z; 
#line 344
}; 
#endif
#line 346 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\vector_types.h"
#if 0
#line 346
struct __declspec(align(16)) double4 { 
#line 348
double x, y, z, w; 
#line 349
}; 
#endif
#line 363 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\vector_types.h"
#if 0
typedef char1 
#line 363
char1; 
#endif
#line 364 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\vector_types.h"
#if 0
typedef uchar1 
#line 364
uchar1; 
#endif
#line 365 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\vector_types.h"
#if 0
typedef char2 
#line 365
char2; 
#endif
#line 366 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\vector_types.h"
#if 0
typedef uchar2 
#line 366
uchar2; 
#endif
#line 367 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\vector_types.h"
#if 0
typedef char3 
#line 367
char3; 
#endif
#line 368 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\vector_types.h"
#if 0
typedef uchar3 
#line 368
uchar3; 
#endif
#line 369 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\vector_types.h"
#if 0
typedef char4 
#line 369
char4; 
#endif
#line 370 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\vector_types.h"
#if 0
typedef uchar4 
#line 370
uchar4; 
#endif
#line 371 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\vector_types.h"
#if 0
typedef short1 
#line 371
short1; 
#endif
#line 372 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\vector_types.h"
#if 0
typedef ushort1 
#line 372
ushort1; 
#endif
#line 373 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\vector_types.h"
#if 0
typedef short2 
#line 373
short2; 
#endif
#line 374 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\vector_types.h"
#if 0
typedef ushort2 
#line 374
ushort2; 
#endif
#line 375 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\vector_types.h"
#if 0
typedef short3 
#line 375
short3; 
#endif
#line 376 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\vector_types.h"
#if 0
typedef ushort3 
#line 376
ushort3; 
#endif
#line 377 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\vector_types.h"
#if 0
typedef short4 
#line 377
short4; 
#endif
#line 378 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\vector_types.h"
#if 0
typedef ushort4 
#line 378
ushort4; 
#endif
#line 379 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\vector_types.h"
#if 0
typedef int1 
#line 379
int1; 
#endif
#line 380 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\vector_types.h"
#if 0
typedef uint1 
#line 380
uint1; 
#endif
#line 381 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\vector_types.h"
#if 0
typedef int2 
#line 381
int2; 
#endif
#line 382 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\vector_types.h"
#if 0
typedef uint2 
#line 382
uint2; 
#endif
#line 383 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\vector_types.h"
#if 0
typedef int3 
#line 383
int3; 
#endif
#line 384 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\vector_types.h"
#if 0
typedef uint3 
#line 384
uint3; 
#endif
#line 385 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\vector_types.h"
#if 0
typedef int4 
#line 385
int4; 
#endif
#line 386 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\vector_types.h"
#if 0
typedef uint4 
#line 386
uint4; 
#endif
#line 387 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\vector_types.h"
#if 0
typedef long1 
#line 387
long1; 
#endif
#line 388 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\vector_types.h"
#if 0
typedef ulong1 
#line 388
ulong1; 
#endif
#line 389 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\vector_types.h"
#if 0
typedef long2 
#line 389
long2; 
#endif
#line 390 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\vector_types.h"
#if 0
typedef ulong2 
#line 390
ulong2; 
#endif
#line 391 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\vector_types.h"
#if 0
typedef long3 
#line 391
long3; 
#endif
#line 392 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\vector_types.h"
#if 0
typedef ulong3 
#line 392
ulong3; 
#endif
#line 393 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\vector_types.h"
#if 0
typedef long4 
#line 393
long4; 
#endif
#line 394 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\vector_types.h"
#if 0
typedef ulong4 
#line 394
ulong4; 
#endif
#line 395 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\vector_types.h"
#if 0
typedef float1 
#line 395
float1; 
#endif
#line 396 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\vector_types.h"
#if 0
typedef float2 
#line 396
float2; 
#endif
#line 397 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\vector_types.h"
#if 0
typedef float3 
#line 397
float3; 
#endif
#line 398 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\vector_types.h"
#if 0
typedef float4 
#line 398
float4; 
#endif
#line 399 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\vector_types.h"
#if 0
typedef longlong1 
#line 399
longlong1; 
#endif
#line 400 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\vector_types.h"
#if 0
typedef ulonglong1 
#line 400
ulonglong1; 
#endif
#line 401 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\vector_types.h"
#if 0
typedef longlong2 
#line 401
longlong2; 
#endif
#line 402 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\vector_types.h"
#if 0
typedef ulonglong2 
#line 402
ulonglong2; 
#endif
#line 403 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\vector_types.h"
#if 0
typedef longlong3 
#line 403
longlong3; 
#endif
#line 404 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\vector_types.h"
#if 0
typedef ulonglong3 
#line 404
ulonglong3; 
#endif
#line 405 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\vector_types.h"
#if 0
typedef longlong4 
#line 405
longlong4; 
#endif
#line 406 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\vector_types.h"
#if 0
typedef ulonglong4 
#line 406
ulonglong4; 
#endif
#line 407 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\vector_types.h"
#if 0
typedef double1 
#line 407
double1; 
#endif
#line 408 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\vector_types.h"
#if 0
typedef double2 
#line 408
double2; 
#endif
#line 409 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\vector_types.h"
#if 0
typedef double3 
#line 409
double3; 
#endif
#line 410 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\vector_types.h"
#if 0
typedef double4 
#line 410
double4; 
#endif
#line 418 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\vector_types.h"
#if 0
#line 418
struct dim3 { 
#line 420
unsigned x, y, z; 
#line 432 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\vector_types.h"
}; 
#endif
#line 434 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\vector_types.h"
#if 0
typedef dim3 
#line 434
dim3; 
#endif
#line 13 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Enterprise\\VC\\Tools\\MSVC\\14.37.32822\\include\\limits.h"
#pragma warning(push)
#pragma warning(disable: 4514 4820 )
#line 16
__pragma( pack ( push, 8 )) extern "C" {
#line 76 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Enterprise\\VC\\Tools\\MSVC\\14.37.32822\\include\\limits.h"
}__pragma( pack ( pop )) 
#line 78
#pragma warning(pop)
#line 14 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\stddef.h"
#pragma warning(push)
#pragma warning(disable: 4324 4514 4574 4710 4793 4820 4995 4996 28719 28726 28727 )
#line 18
__pragma( pack ( push, 8 )) extern "C" {
#line 23
namespace std { 
#line 25
typedef decltype(nullptr) nullptr_t; 
#line 26
}
#line 28
using std::nullptr_t;
#line 35 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\stddef.h"
int *__cdecl _errno(); 
#line 38
errno_t __cdecl _set_errno(int _Value); 
#line 39
errno_t __cdecl _get_errno(int * _Value); 
#line 55 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\stddef.h"
extern unsigned long __cdecl __threadid(); 
#line 57
extern uintptr_t __cdecl __threadhandle(); 
#line 61
}__pragma( pack ( pop )) 
#line 63
#pragma warning(pop)
#line 202 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\driver_types.h"
#if 0
#line 202
enum cudaError { 
#line 209
cudaSuccess, 
#line 215
cudaErrorInvalidValue, 
#line 221
cudaErrorMemoryAllocation, 
#line 227
cudaErrorInitializationError, 
#line 234
cudaErrorCudartUnloading, 
#line 241
cudaErrorProfilerDisabled, 
#line 249
cudaErrorProfilerNotInitialized, 
#line 256
cudaErrorProfilerAlreadyStarted, 
#line 263
cudaErrorProfilerAlreadyStopped, 
#line 272
cudaErrorInvalidConfiguration, 
#line 278
cudaErrorInvalidPitchValue = 12, 
#line 284
cudaErrorInvalidSymbol, 
#line 292
cudaErrorInvalidHostPointer = 16, 
#line 300
cudaErrorInvalidDevicePointer, 
#line 306
cudaErrorInvalidTexture, 
#line 312
cudaErrorInvalidTextureBinding, 
#line 319
cudaErrorInvalidChannelDescriptor, 
#line 325
cudaErrorInvalidMemcpyDirection, 
#line 335
cudaErrorAddressOfConstant, 
#line 344
cudaErrorTextureFetchFailed, 
#line 353
cudaErrorTextureNotBound, 
#line 362
cudaErrorSynchronizationError, 
#line 368
cudaErrorInvalidFilterSetting, 
#line 374
cudaErrorInvalidNormSetting, 
#line 382
cudaErrorMixedDeviceExecution, 
#line 390
cudaErrorNotYetImplemented = 31, 
#line 399
cudaErrorMemoryValueTooLarge, 
#line 406
cudaErrorStubLibrary = 34, 
#line 413
cudaErrorInsufficientDriver, 
#line 420
cudaErrorCallRequiresNewerDriver, 
#line 426
cudaErrorInvalidSurface, 
#line 432
cudaErrorDuplicateVariableName = 43, 
#line 438
cudaErrorDuplicateTextureName, 
#line 444
cudaErrorDuplicateSurfaceName, 
#line 454
cudaErrorDevicesUnavailable, 
#line 467
cudaErrorIncompatibleDriverContext = 49, 
#line 473
cudaErrorMissingConfiguration = 52, 
#line 482
cudaErrorPriorLaunchFailure, 
#line 489
cudaErrorLaunchMaxDepthExceeded = 65, 
#line 497
cudaErrorLaunchFileScopedTex, 
#line 505
cudaErrorLaunchFileScopedSurf, 
#line 520
cudaErrorSyncDepthExceeded, 
#line 532
cudaErrorLaunchPendingCountExceeded, 
#line 538
cudaErrorInvalidDeviceFunction = 98, 
#line 544
cudaErrorNoDevice = 100, 
#line 551
cudaErrorInvalidDevice, 
#line 556
cudaErrorDeviceNotLicensed, 
#line 565
cudaErrorSoftwareValidityNotEstablished, 
#line 570
cudaErrorStartupFailure = 127, 
#line 575
cudaErrorInvalidKernelImage = 200, 
#line 585
cudaErrorDeviceUninitialized, 
#line 590
cudaErrorMapBufferObjectFailed = 205, 
#line 595
cudaErrorUnmapBufferObjectFailed, 
#line 601
cudaErrorArrayIsMapped, 
#line 606
cudaErrorAlreadyMapped, 
#line 614
cudaErrorNoKernelImageForDevice, 
#line 619
cudaErrorAlreadyAcquired, 
#line 624
cudaErrorNotMapped, 
#line 630
cudaErrorNotMappedAsArray, 
#line 636
cudaErrorNotMappedAsPointer, 
#line 642
cudaErrorECCUncorrectable, 
#line 648
cudaErrorUnsupportedLimit, 
#line 654
cudaErrorDeviceAlreadyInUse, 
#line 660
cudaErrorPeerAccessUnsupported, 
#line 666
cudaErrorInvalidPtx, 
#line 671
cudaErrorInvalidGraphicsContext, 
#line 677
cudaErrorNvlinkUncorrectable, 
#line 684
cudaErrorJitCompilerNotFound, 
#line 691
cudaErrorUnsupportedPtxVersion, 
#line 698
cudaErrorJitCompilationDisabled, 
#line 703
cudaErrorUnsupportedExecAffinity, 
#line 708
cudaErrorInvalidSource = 300, 
#line 713
cudaErrorFileNotFound, 
#line 718
cudaErrorSharedObjectSymbolNotFound, 
#line 723
cudaErrorSharedObjectInitFailed, 
#line 728
cudaErrorOperatingSystem, 
#line 735
cudaErrorInvalidResourceHandle = 400, 
#line 741
cudaErrorIllegalState, 
#line 748
cudaErrorSymbolNotFound = 500, 
#line 756
cudaErrorNotReady = 600, 
#line 764
cudaErrorIllegalAddress = 700, 
#line 773
cudaErrorLaunchOutOfResources, 
#line 784
cudaErrorLaunchTimeout, 
#line 790
cudaErrorLaunchIncompatibleTexturing, 
#line 797
cudaErrorPeerAccessAlreadyEnabled, 
#line 804
cudaErrorPeerAccessNotEnabled, 
#line 817
cudaErrorSetOnActiveProcess = 708, 
#line 824
cudaErrorContextIsDestroyed, 
#line 831
cudaErrorAssert, 
#line 838
cudaErrorTooManyPeers, 
#line 844
cudaErrorHostMemoryAlreadyRegistered, 
#line 850
cudaErrorHostMemoryNotRegistered, 
#line 859
cudaErrorHardwareStackError, 
#line 867
cudaErrorIllegalInstruction, 
#line 876
cudaErrorMisalignedAddress, 
#line 887
cudaErrorInvalidAddressSpace, 
#line 895
cudaErrorInvalidPc, 
#line 906
cudaErrorLaunchFailure, 
#line 915
cudaErrorCooperativeLaunchTooLarge, 
#line 920
cudaErrorNotPermitted = 800, 
#line 926
cudaErrorNotSupported, 
#line 935
cudaErrorSystemNotReady, 
#line 942
cudaErrorSystemDriverMismatch, 
#line 951
cudaErrorCompatNotSupportedOnDevice, 
#line 956
cudaErrorMpsConnectionFailed, 
#line 961
cudaErrorMpsRpcFailure, 
#line 967
cudaErrorMpsServerNotReady, 
#line 972
cudaErrorMpsMaxClientsReached, 
#line 977
cudaErrorMpsMaxConnectionsReached, 
#line 982
cudaErrorMpsClientTerminated, 
#line 987
cudaErrorStreamCaptureUnsupported = 900, 
#line 993
cudaErrorStreamCaptureInvalidated, 
#line 999
cudaErrorStreamCaptureMerge, 
#line 1004
cudaErrorStreamCaptureUnmatched, 
#line 1010
cudaErrorStreamCaptureUnjoined, 
#line 1017
cudaErrorStreamCaptureIsolation, 
#line 1023
cudaErrorStreamCaptureImplicit, 
#line 1029
cudaErrorCapturedEvent, 
#line 1036
cudaErrorStreamCaptureWrongThread, 
#line 1041
cudaErrorTimeout, 
#line 1047
cudaErrorGraphExecUpdateFailure, 
#line 1057
cudaErrorExternalDevice, 
#line 1063
cudaErrorInvalidClusterSize, 
#line 1068
cudaErrorUnknown = 999, 
#line 1076
cudaErrorApiFailureBase = 10000
#line 1077
}; 
#endif
#line 1082 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\driver_types.h"
#if 0
#line 1082
enum cudaChannelFormatKind { 
#line 1084
cudaChannelFormatKindSigned, 
#line 1085
cudaChannelFormatKindUnsigned, 
#line 1086
cudaChannelFormatKindFloat, 
#line 1087
cudaChannelFormatKindNone, 
#line 1088
cudaChannelFormatKindNV12, 
#line 1089
cudaChannelFormatKindUnsignedNormalized8X1, 
#line 1090
cudaChannelFormatKindUnsignedNormalized8X2, 
#line 1091
cudaChannelFormatKindUnsignedNormalized8X4, 
#line 1092
cudaChannelFormatKindUnsignedNormalized16X1, 
#line 1093
cudaChannelFormatKindUnsignedNormalized16X2, 
#line 1094
cudaChannelFormatKindUnsignedNormalized16X4, 
#line 1095
cudaChannelFormatKindSignedNormalized8X1, 
#line 1096
cudaChannelFormatKindSignedNormalized8X2, 
#line 1097
cudaChannelFormatKindSignedNormalized8X4, 
#line 1098
cudaChannelFormatKindSignedNormalized16X1, 
#line 1099
cudaChannelFormatKindSignedNormalized16X2, 
#line 1100
cudaChannelFormatKindSignedNormalized16X4, 
#line 1101
cudaChannelFormatKindUnsignedBlockCompressed1, 
#line 1102
cudaChannelFormatKindUnsignedBlockCompressed1SRGB, 
#line 1103
cudaChannelFormatKindUnsignedBlockCompressed2, 
#line 1104
cudaChannelFormatKindUnsignedBlockCompressed2SRGB, 
#line 1105
cudaChannelFormatKindUnsignedBlockCompressed3, 
#line 1106
cudaChannelFormatKindUnsignedBlockCompressed3SRGB, 
#line 1107
cudaChannelFormatKindUnsignedBlockCompressed4, 
#line 1108
cudaChannelFormatKindSignedBlockCompressed4, 
#line 1109
cudaChannelFormatKindUnsignedBlockCompressed5, 
#line 1110
cudaChannelFormatKindSignedBlockCompressed5, 
#line 1111
cudaChannelFormatKindUnsignedBlockCompressed6H, 
#line 1112
cudaChannelFormatKindSignedBlockCompressed6H, 
#line 1113
cudaChannelFormatKindUnsignedBlockCompressed7, 
#line 1114
cudaChannelFormatKindUnsignedBlockCompressed7SRGB
#line 1115
}; 
#endif
#line 1120 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\driver_types.h"
#if 0
#line 1120
struct cudaChannelFormatDesc { 
#line 1122
int x; 
#line 1123
int y; 
#line 1124
int z; 
#line 1125
int w; 
#line 1126
cudaChannelFormatKind f; 
#line 1127
}; 
#endif
#line 1132 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\driver_types.h"
typedef struct cudaArray *cudaArray_t; 
#line 1137
typedef const cudaArray *cudaArray_const_t; 
#line 1139
struct cudaArray; 
#line 1144
typedef struct cudaMipmappedArray *cudaMipmappedArray_t; 
#line 1149
typedef const cudaMipmappedArray *cudaMipmappedArray_const_t; 
#line 1151
struct cudaMipmappedArray; 
#line 1161
#if 0
#line 1161
struct cudaArraySparseProperties { 
#line 1162
struct { 
#line 1163
unsigned width; 
#line 1164
unsigned height; 
#line 1165
unsigned depth; 
#line 1166
} tileExtent; 
#line 1167
unsigned miptailFirstLevel; 
#line 1168
unsigned __int64 miptailSize; 
#line 1169
unsigned flags; 
#line 1170
unsigned reserved[4]; 
#line 1171
}; 
#endif
#line 1176 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\driver_types.h"
#if 0
#line 1176
struct cudaArrayMemoryRequirements { 
#line 1177
size_t size; 
#line 1178
size_t alignment; 
#line 1179
unsigned reserved[4]; 
#line 1180
}; 
#endif
#line 1185 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\driver_types.h"
#if 0
#line 1185
enum cudaMemoryType { 
#line 1187
cudaMemoryTypeUnregistered, 
#line 1188
cudaMemoryTypeHost, 
#line 1189
cudaMemoryTypeDevice, 
#line 1190
cudaMemoryTypeManaged
#line 1191
}; 
#endif
#line 1196 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\driver_types.h"
#if 0
#line 1196
enum cudaMemcpyKind { 
#line 1198
cudaMemcpyHostToHost, 
#line 1199
cudaMemcpyHostToDevice, 
#line 1200
cudaMemcpyDeviceToHost, 
#line 1201
cudaMemcpyDeviceToDevice, 
#line 1202
cudaMemcpyDefault
#line 1203
}; 
#endif
#line 1210 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\driver_types.h"
#if 0
#line 1210
struct cudaPitchedPtr { 
#line 1212
void *ptr; 
#line 1213
size_t pitch; 
#line 1214
size_t xsize; 
#line 1215
size_t ysize; 
#line 1216
}; 
#endif
#line 1223 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\driver_types.h"
#if 0
#line 1223
struct cudaExtent { 
#line 1225
size_t width; 
#line 1226
size_t height; 
#line 1227
size_t depth; 
#line 1228
}; 
#endif
#line 1235 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\driver_types.h"
#if 0
#line 1235
struct cudaPos { 
#line 1237
size_t x; 
#line 1238
size_t y; 
#line 1239
size_t z; 
#line 1240
}; 
#endif
#line 1245 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\driver_types.h"
#if 0
#line 1245
struct cudaMemcpy3DParms { 
#line 1247
cudaArray_t srcArray; 
#line 1248
cudaPos srcPos; 
#line 1249
cudaPitchedPtr srcPtr; 
#line 1251
cudaArray_t dstArray; 
#line 1252
cudaPos dstPos; 
#line 1253
cudaPitchedPtr dstPtr; 
#line 1255
cudaExtent extent; 
#line 1256
cudaMemcpyKind kind; __pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)
#line 1257
}; 
#endif
#line 1262 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\driver_types.h"
#if 0
#line 1262
struct cudaMemcpy3DPeerParms { 
#line 1264
cudaArray_t srcArray; 
#line 1265
cudaPos srcPos; 
#line 1266
cudaPitchedPtr srcPtr; 
#line 1267
int srcDevice; 
#line 1269
cudaArray_t dstArray; 
#line 1270
cudaPos dstPos; 
#line 1271
cudaPitchedPtr dstPtr; 
#line 1272
int dstDevice; 
#line 1274
cudaExtent extent; 
#line 1275
}; 
#endif
#line 1280 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\driver_types.h"
#if 0
#line 1280
struct cudaMemsetParams { 
#line 1281
void *dst; 
#line 1282
size_t pitch; 
#line 1283
unsigned value; 
#line 1284
unsigned elementSize; 
#line 1285
size_t width; 
#line 1286
size_t height; 
#line 1287
}; 
#endif
#line 1292 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\driver_types.h"
#if 0
#line 1292
enum cudaAccessProperty { 
#line 1293
cudaAccessPropertyNormal, 
#line 1294
cudaAccessPropertyStreaming, 
#line 1295
cudaAccessPropertyPersisting
#line 1296
}; 
#endif
#line 1309 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\driver_types.h"
#if 0
#line 1309
struct cudaAccessPolicyWindow { 
#line 1310
void *base_ptr; 
#line 1311
size_t num_bytes; 
#line 1312
float hitRatio; 
#line 1313
cudaAccessProperty hitProp; 
#line 1314
cudaAccessProperty missProp; __pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)
#line 1315
}; 
#endif
#line 1327 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\driver_types.h"
typedef void (__stdcall *cudaHostFn_t)(void * userData); 
#line 1332
#if 0
#line 1332
struct cudaHostNodeParams { 
#line 1333
cudaHostFn_t fn; 
#line 1334
void *userData; 
#line 1335
}; 
#endif
#line 1340 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\driver_types.h"
#if 0
#line 1340
enum cudaStreamCaptureStatus { 
#line 1341
cudaStreamCaptureStatusNone, 
#line 1342
cudaStreamCaptureStatusActive, 
#line 1343
cudaStreamCaptureStatusInvalidated
#line 1345
}; 
#endif
#line 1351 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\driver_types.h"
#if 0
#line 1351
enum cudaStreamCaptureMode { 
#line 1352
cudaStreamCaptureModeGlobal, 
#line 1353
cudaStreamCaptureModeThreadLocal, 
#line 1354
cudaStreamCaptureModeRelaxed
#line 1355
}; 
#endif
#line 1357 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\driver_types.h"
#if 0
#line 1357
enum cudaSynchronizationPolicy { 
#line 1358
cudaSyncPolicyAuto = 1, 
#line 1359
cudaSyncPolicySpin, 
#line 1360
cudaSyncPolicyYield, 
#line 1361
cudaSyncPolicyBlockingSync
#line 1362
}; 
#endif
#line 1367 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\driver_types.h"
#if 0
#line 1367
enum cudaClusterSchedulingPolicy { 
#line 1368
cudaClusterSchedulingPolicyDefault, 
#line 1369
cudaClusterSchedulingPolicySpread, 
#line 1370
cudaClusterSchedulingPolicyLoadBalancing
#line 1371
}; 
#endif
#line 1376 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\driver_types.h"
#if 0
#line 1376
enum cudaStreamUpdateCaptureDependenciesFlags { 
#line 1377
cudaStreamAddCaptureDependencies, 
#line 1378
cudaStreamSetCaptureDependencies
#line 1379
}; 
#endif
#line 1384 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\driver_types.h"
#if 0
#line 1384
enum cudaUserObjectFlags { 
#line 1385
cudaUserObjectNoDestructorSync = 1
#line 1386
}; 
#endif
#line 1391 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\driver_types.h"
#if 0
#line 1391
enum cudaUserObjectRetainFlags { 
#line 1392
cudaGraphUserObjectMove = 1
#line 1393
}; 
#endif
#line 1398 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\driver_types.h"
struct cudaGraphicsResource; 
#line 1403
#if 0
#line 1403
enum cudaGraphicsRegisterFlags { 
#line 1405
cudaGraphicsRegisterFlagsNone, 
#line 1406
cudaGraphicsRegisterFlagsReadOnly, 
#line 1407
cudaGraphicsRegisterFlagsWriteDiscard, 
#line 1408
cudaGraphicsRegisterFlagsSurfaceLoadStore = 4, 
#line 1409
cudaGraphicsRegisterFlagsTextureGather = 8
#line 1410
}; 
#endif
#line 1415 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\driver_types.h"
#if 0
#line 1415
enum cudaGraphicsMapFlags { 
#line 1417
cudaGraphicsMapFlagsNone, 
#line 1418
cudaGraphicsMapFlagsReadOnly, 
#line 1419
cudaGraphicsMapFlagsWriteDiscard
#line 1420
}; 
#endif
#line 1425 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\driver_types.h"
#if 0
#line 1425
enum cudaGraphicsCubeFace { 
#line 1427
cudaGraphicsCubeFacePositiveX, 
#line 1428
cudaGraphicsCubeFaceNegativeX, 
#line 1429
cudaGraphicsCubeFacePositiveY, 
#line 1430
cudaGraphicsCubeFaceNegativeY, 
#line 1431
cudaGraphicsCubeFacePositiveZ, 
#line 1432
cudaGraphicsCubeFaceNegativeZ
#line 1433
}; 
#endif
#line 1438 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\driver_types.h"
#if 0
#line 1438
enum cudaResourceType { 
#line 1440
cudaResourceTypeArray, 
#line 1441
cudaResourceTypeMipmappedArray, 
#line 1442
cudaResourceTypeLinear, 
#line 1443
cudaResourceTypePitch2D
#line 1444
}; 
#endif
#line 1449 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\driver_types.h"
#if 0
#line 1449
enum cudaResourceViewFormat { 
#line 1451
cudaResViewFormatNone, 
#line 1452
cudaResViewFormatUnsignedChar1, 
#line 1453
cudaResViewFormatUnsignedChar2, 
#line 1454
cudaResViewFormatUnsignedChar4, 
#line 1455
cudaResViewFormatSignedChar1, 
#line 1456
cudaResViewFormatSignedChar2, 
#line 1457
cudaResViewFormatSignedChar4, 
#line 1458
cudaResViewFormatUnsignedShort1, 
#line 1459
cudaResViewFormatUnsignedShort2, 
#line 1460
cudaResViewFormatUnsignedShort4, 
#line 1461
cudaResViewFormatSignedShort1, 
#line 1462
cudaResViewFormatSignedShort2, 
#line 1463
cudaResViewFormatSignedShort4, 
#line 1464
cudaResViewFormatUnsignedInt1, 
#line 1465
cudaResViewFormatUnsignedInt2, 
#line 1466
cudaResViewFormatUnsignedInt4, 
#line 1467
cudaResViewFormatSignedInt1, 
#line 1468
cudaResViewFormatSignedInt2, 
#line 1469
cudaResViewFormatSignedInt4, 
#line 1470
cudaResViewFormatHalf1, 
#line 1471
cudaResViewFormatHalf2, 
#line 1472
cudaResViewFormatHalf4, 
#line 1473
cudaResViewFormatFloat1, 
#line 1474
cudaResViewFormatFloat2, 
#line 1475
cudaResViewFormatFloat4, 
#line 1476
cudaResViewFormatUnsignedBlockCompressed1, 
#line 1477
cudaResViewFormatUnsignedBlockCompressed2, 
#line 1478
cudaResViewFormatUnsignedBlockCompressed3, 
#line 1479
cudaResViewFormatUnsignedBlockCompressed4, 
#line 1480
cudaResViewFormatSignedBlockCompressed4, 
#line 1481
cudaResViewFormatUnsignedBlockCompressed5, 
#line 1482
cudaResViewFormatSignedBlockCompressed5, 
#line 1483
cudaResViewFormatUnsignedBlockCompressed6H, 
#line 1484
cudaResViewFormatSignedBlockCompressed6H, 
#line 1485
cudaResViewFormatUnsignedBlockCompressed7
#line 1486
}; 
#endif
#line 1491 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\driver_types.h"
#if 0
#line 1491
struct cudaResourceDesc { 
#line 1492
cudaResourceType resType; 
#line 1494
union { 
#line 1495
struct { 
#line 1496
cudaArray_t array; 
#line 1497
} array; 
#line 1498
struct { 
#line 1499
cudaMipmappedArray_t mipmap; 
#line 1500
} mipmap; 
#line 1501
struct { 
#line 1502
void *devPtr; 
#line 1503
cudaChannelFormatDesc desc; 
#line 1504
size_t sizeInBytes; 
#line 1505
} linear; 
#line 1506
struct { 
#line 1507
void *devPtr; 
#line 1508
cudaChannelFormatDesc desc; 
#line 1509
size_t width; 
#line 1510
size_t height; 
#line 1511
size_t pitchInBytes; 
#line 1512
} pitch2D; 
#line 1513
} res; 
#line 1514
}; 
#endif
#line 1519 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\driver_types.h"
#if 0
#line 1519
struct cudaResourceViewDesc { 
#line 1521
cudaResourceViewFormat format; 
#line 1522
size_t width; 
#line 1523
size_t height; 
#line 1524
size_t depth; 
#line 1525
unsigned firstMipmapLevel; 
#line 1526
unsigned lastMipmapLevel; 
#line 1527
unsigned firstLayer; 
#line 1528
unsigned lastLayer; 
#line 1529
}; 
#endif
#line 1534 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\driver_types.h"
#if 0
#line 1534
struct cudaPointerAttributes { 
#line 1540
cudaMemoryType type; 
#line 1551
int device; 
#line 1557
void *devicePointer; 
#line 1566
void *hostPointer; 
#line 1567
}; 
#endif
#line 1572 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\driver_types.h"
#if 0
#line 1572
struct cudaFuncAttributes { 
#line 1579
size_t sharedSizeBytes; 
#line 1585
size_t constSizeBytes; 
#line 1590
size_t localSizeBytes; 
#line 1597
int maxThreadsPerBlock; 
#line 1602
int numRegs; 
#line 1609
int ptxVersion; 
#line 1616
int binaryVersion; 
#line 1622
int cacheModeCA; 
#line 1629
int maxDynamicSharedSizeBytes; 
#line 1638
int preferredShmemCarveout; 
#line 1639
}; 
#endif
#line 1644 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\driver_types.h"
#if 0
#line 1644
enum cudaFuncAttribute { 
#line 1646
cudaFuncAttributeMaxDynamicSharedMemorySize = 8, 
#line 1647
cudaFuncAttributePreferredSharedMemoryCarveout, 
#line 1648
cudaFuncAttributeClusterDimMustBeSet, 
#line 1649
cudaFuncAttributeRequiredClusterWidth, 
#line 1650
cudaFuncAttributeRequiredClusterHeight, 
#line 1651
cudaFuncAttributeRequiredClusterDepth, 
#line 1652
cudaFuncAttributeNonPortableClusterSizeAllowed, 
#line 1653
cudaFuncAttributeClusterSchedulingPolicyPreference, 
#line 1654
cudaFuncAttributeMax
#line 1655
}; 
#endif
#line 1660 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\driver_types.h"
#if 0
#line 1660
enum cudaFuncCache { 
#line 1662
cudaFuncCachePreferNone, 
#line 1663
cudaFuncCachePreferShared, 
#line 1664
cudaFuncCachePreferL1, 
#line 1665
cudaFuncCachePreferEqual
#line 1666
}; 
#endif
#line 1672 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\driver_types.h"
#if 0
#line 1672
enum cudaSharedMemConfig { 
#line 1674
cudaSharedMemBankSizeDefault, 
#line 1675
cudaSharedMemBankSizeFourByte, 
#line 1676
cudaSharedMemBankSizeEightByte
#line 1677
}; 
#endif
#line 1682 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\driver_types.h"
#if 0
#line 1682
enum cudaSharedCarveout { 
#line 1683
cudaSharedmemCarveoutDefault = (-1), 
#line 1684
cudaSharedmemCarveoutMaxShared = 100, 
#line 1685
cudaSharedmemCarveoutMaxL1 = 0
#line 1686
}; 
#endif
#line 1691 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\driver_types.h"
#if 0
#line 1691
enum cudaComputeMode { 
#line 1693
cudaComputeModeDefault, 
#line 1694
cudaComputeModeExclusive, 
#line 1695
cudaComputeModeProhibited, 
#line 1696
cudaComputeModeExclusiveProcess
#line 1697
}; 
#endif
#line 1702 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\driver_types.h"
#if 0
#line 1702
enum cudaLimit { 
#line 1704
cudaLimitStackSize, 
#line 1705
cudaLimitPrintfFifoSize, 
#line 1706
cudaLimitMallocHeapSize, 
#line 1707
cudaLimitDevRuntimeSyncDepth, 
#line 1708
cudaLimitDevRuntimePendingLaunchCount, 
#line 1709
cudaLimitMaxL2FetchGranularity, 
#line 1710
cudaLimitPersistingL2CacheSize
#line 1711
}; 
#endif
#line 1716 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\driver_types.h"
#if 0
#line 1716
enum cudaMemoryAdvise { 
#line 1718
cudaMemAdviseSetReadMostly = 1, 
#line 1719
cudaMemAdviseUnsetReadMostly, 
#line 1720
cudaMemAdviseSetPreferredLocation, 
#line 1721
cudaMemAdviseUnsetPreferredLocation, 
#line 1722
cudaMemAdviseSetAccessedBy, 
#line 1723
cudaMemAdviseUnsetAccessedBy
#line 1724
}; 
#endif
#line 1729 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\driver_types.h"
#if 0
#line 1729
enum cudaMemRangeAttribute { 
#line 1731
cudaMemRangeAttributeReadMostly = 1, 
#line 1732
cudaMemRangeAttributePreferredLocation, 
#line 1733
cudaMemRangeAttributeAccessedBy, 
#line 1734
cudaMemRangeAttributeLastPrefetchLocation
#line 1735
}; 
#endif
#line 1740 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\driver_types.h"
#if 0
#line 1740
enum cudaOutputMode { 
#line 1742
cudaKeyValuePair, 
#line 1743
cudaCSV
#line 1744
}; 
#endif
#line 1749 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\driver_types.h"
#if 0
#line 1749
enum cudaFlushGPUDirectRDMAWritesOptions { 
#line 1750
cudaFlushGPUDirectRDMAWritesOptionHost = (1 << 0), 
#line 1751
cudaFlushGPUDirectRDMAWritesOptionMemOps
#line 1752
}; 
#endif
#line 1757 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\driver_types.h"
#if 0
#line 1757
enum cudaGPUDirectRDMAWritesOrdering { 
#line 1758
cudaGPUDirectRDMAWritesOrderingNone, 
#line 1759
cudaGPUDirectRDMAWritesOrderingOwner = 100, 
#line 1760
cudaGPUDirectRDMAWritesOrderingAllDevices = 200
#line 1761
}; 
#endif
#line 1766 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\driver_types.h"
#if 0
#line 1766
enum cudaFlushGPUDirectRDMAWritesScope { 
#line 1767
cudaFlushGPUDirectRDMAWritesToOwner = 100, 
#line 1768
cudaFlushGPUDirectRDMAWritesToAllDevices = 200
#line 1769
}; 
#endif
#line 1774 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\driver_types.h"
#if 0
#line 1774
enum cudaFlushGPUDirectRDMAWritesTarget { 
#line 1775
cudaFlushGPUDirectRDMAWritesTargetCurrentDevice
#line 1776
}; 
#endif
#line 1782 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\driver_types.h"
#if 0
#line 1782
enum cudaDeviceAttr { 
#line 1784
cudaDevAttrMaxThreadsPerBlock = 1, 
#line 1785
cudaDevAttrMaxBlockDimX, 
#line 1786
cudaDevAttrMaxBlockDimY, 
#line 1787
cudaDevAttrMaxBlockDimZ, 
#line 1788
cudaDevAttrMaxGridDimX, 
#line 1789
cudaDevAttrMaxGridDimY, 
#line 1790
cudaDevAttrMaxGridDimZ, 
#line 1791
cudaDevAttrMaxSharedMemoryPerBlock, 
#line 1792
cudaDevAttrTotalConstantMemory, 
#line 1793
cudaDevAttrWarpSize, 
#line 1794
cudaDevAttrMaxPitch, 
#line 1795
cudaDevAttrMaxRegistersPerBlock, 
#line 1796
cudaDevAttrClockRate, 
#line 1797
cudaDevAttrTextureAlignment, 
#line 1798
cudaDevAttrGpuOverlap, 
#line 1799
cudaDevAttrMultiProcessorCount, 
#line 1800
cudaDevAttrKernelExecTimeout, 
#line 1801
cudaDevAttrIntegrated, 
#line 1802
cudaDevAttrCanMapHostMemory, 
#line 1803
cudaDevAttrComputeMode, 
#line 1804
cudaDevAttrMaxTexture1DWidth, 
#line 1805
cudaDevAttrMaxTexture2DWidth, 
#line 1806
cudaDevAttrMaxTexture2DHeight, 
#line 1807
cudaDevAttrMaxTexture3DWidth, 
#line 1808
cudaDevAttrMaxTexture3DHeight, 
#line 1809
cudaDevAttrMaxTexture3DDepth, 
#line 1810
cudaDevAttrMaxTexture2DLayeredWidth, 
#line 1811
cudaDevAttrMaxTexture2DLayeredHeight, 
#line 1812
cudaDevAttrMaxTexture2DLayeredLayers, 
#line 1813
cudaDevAttrSurfaceAlignment, 
#line 1814
cudaDevAttrConcurrentKernels, 
#line 1815
cudaDevAttrEccEnabled, 
#line 1816
cudaDevAttrPciBusId, 
#line 1817
cudaDevAttrPciDeviceId, 
#line 1818
cudaDevAttrTccDriver, 
#line 1819
cudaDevAttrMemoryClockRate, 
#line 1820
cudaDevAttrGlobalMemoryBusWidth, 
#line 1821
cudaDevAttrL2CacheSize, 
#line 1822
cudaDevAttrMaxThreadsPerMultiProcessor, 
#line 1823
cudaDevAttrAsyncEngineCount, 
#line 1824
cudaDevAttrUnifiedAddressing, 
#line 1825
cudaDevAttrMaxTexture1DLayeredWidth, 
#line 1826
cudaDevAttrMaxTexture1DLayeredLayers, 
#line 1827
cudaDevAttrMaxTexture2DGatherWidth = 45, 
#line 1828
cudaDevAttrMaxTexture2DGatherHeight, 
#line 1829
cudaDevAttrMaxTexture3DWidthAlt, 
#line 1830
cudaDevAttrMaxTexture3DHeightAlt, 
#line 1831
cudaDevAttrMaxTexture3DDepthAlt, 
#line 1832
cudaDevAttrPciDomainId, 
#line 1833
cudaDevAttrTexturePitchAlignment, 
#line 1834
cudaDevAttrMaxTextureCubemapWidth, 
#line 1835
cudaDevAttrMaxTextureCubemapLayeredWidth, 
#line 1836
cudaDevAttrMaxTextureCubemapLayeredLayers, 
#line 1837
cudaDevAttrMaxSurface1DWidth, 
#line 1838
cudaDevAttrMaxSurface2DWidth, 
#line 1839
cudaDevAttrMaxSurface2DHeight, 
#line 1840
cudaDevAttrMaxSurface3DWidth, 
#line 1841
cudaDevAttrMaxSurface3DHeight, 
#line 1842
cudaDevAttrMaxSurface3DDepth, 
#line 1843
cudaDevAttrMaxSurface1DLayeredWidth, 
#line 1844
cudaDevAttrMaxSurface1DLayeredLayers, 
#line 1845
cudaDevAttrMaxSurface2DLayeredWidth, 
#line 1846
cudaDevAttrMaxSurface2DLayeredHeight, 
#line 1847
cudaDevAttrMaxSurface2DLayeredLayers, 
#line 1848
cudaDevAttrMaxSurfaceCubemapWidth, 
#line 1849
cudaDevAttrMaxSurfaceCubemapLayeredWidth, 
#line 1850
cudaDevAttrMaxSurfaceCubemapLayeredLayers, 
#line 1851
cudaDevAttrMaxTexture1DLinearWidth, 
#line 1852
cudaDevAttrMaxTexture2DLinearWidth, 
#line 1853
cudaDevAttrMaxTexture2DLinearHeight, 
#line 1854
cudaDevAttrMaxTexture2DLinearPitch, 
#line 1855
cudaDevAttrMaxTexture2DMipmappedWidth, 
#line 1856
cudaDevAttrMaxTexture2DMipmappedHeight, 
#line 1857
cudaDevAttrComputeCapabilityMajor, 
#line 1858
cudaDevAttrComputeCapabilityMinor, 
#line 1859
cudaDevAttrMaxTexture1DMipmappedWidth, 
#line 1860
cudaDevAttrStreamPrioritiesSupported, 
#line 1861
cudaDevAttrGlobalL1CacheSupported, 
#line 1862
cudaDevAttrLocalL1CacheSupported, 
#line 1863
cudaDevAttrMaxSharedMemoryPerMultiprocessor, 
#line 1864
cudaDevAttrMaxRegistersPerMultiprocessor, 
#line 1865
cudaDevAttrManagedMemory, 
#line 1866
cudaDevAttrIsMultiGpuBoard, 
#line 1867
cudaDevAttrMultiGpuBoardGroupID, 
#line 1868
cudaDevAttrHostNativeAtomicSupported, 
#line 1869
cudaDevAttrSingleToDoublePrecisionPerfRatio, 
#line 1870
cudaDevAttrPageableMemoryAccess, 
#line 1871
cudaDevAttrConcurrentManagedAccess, 
#line 1872
cudaDevAttrComputePreemptionSupported, 
#line 1873
cudaDevAttrCanUseHostPointerForRegisteredMem, 
#line 1874
cudaDevAttrReserved92, 
#line 1875
cudaDevAttrReserved93, 
#line 1876
cudaDevAttrReserved94, 
#line 1877
cudaDevAttrCooperativeLaunch, 
#line 1878
cudaDevAttrCooperativeMultiDeviceLaunch, 
#line 1879
cudaDevAttrMaxSharedMemoryPerBlockOptin, 
#line 1880
cudaDevAttrCanFlushRemoteWrites, 
#line 1881
cudaDevAttrHostRegisterSupported, 
#line 1882
cudaDevAttrPageableMemoryAccessUsesHostPageTables, 
#line 1883
cudaDevAttrDirectManagedMemAccessFromHost, 
#line 1884
cudaDevAttrMaxBlocksPerMultiprocessor = 106, 
#line 1885
cudaDevAttrMaxPersistingL2CacheSize = 108, 
#line 1886
cudaDevAttrMaxAccessPolicyWindowSize, 
#line 1887
cudaDevAttrReservedSharedMemoryPerBlock = 111, 
#line 1888
cudaDevAttrSparseCudaArraySupported, 
#line 1889
cudaDevAttrHostRegisterReadOnlySupported, 
#line 1890
cudaDevAttrTimelineSemaphoreInteropSupported, 
#line 1891
cudaDevAttrMaxTimelineSemaphoreInteropSupported = 114, 
#line 1892
cudaDevAttrMemoryPoolsSupported, 
#line 1893
cudaDevAttrGPUDirectRDMASupported, 
#line 1894
cudaDevAttrGPUDirectRDMAFlushWritesOptions, 
#line 1895
cudaDevAttrGPUDirectRDMAWritesOrdering, 
#line 1896
cudaDevAttrMemoryPoolSupportedHandleTypes, 
#line 1897
cudaDevAttrClusterLaunch, 
#line 1898
cudaDevAttrDeferredMappingCudaArraySupported, 
#line 1899
cudaDevAttrMax
#line 1900
}; 
#endif
#line 1905 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\driver_types.h"
#if 0
#line 1905
enum cudaMemPoolAttr { 
#line 1915
cudaMemPoolReuseFollowEventDependencies = 1, 
#line 1922
cudaMemPoolReuseAllowOpportunistic, 
#line 1930
cudaMemPoolReuseAllowInternalDependencies, 
#line 1941
cudaMemPoolAttrReleaseThreshold, 
#line 1947
cudaMemPoolAttrReservedMemCurrent, 
#line 1954
cudaMemPoolAttrReservedMemHigh, 
#line 1960
cudaMemPoolAttrUsedMemCurrent, 
#line 1967
cudaMemPoolAttrUsedMemHigh
#line 1968
}; 
#endif
#line 1973 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\driver_types.h"
#if 0
#line 1973
enum cudaMemLocationType { 
#line 1974
cudaMemLocationTypeInvalid, 
#line 1975
cudaMemLocationTypeDevice
#line 1976
}; 
#endif
#line 1983 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\driver_types.h"
#if 0
#line 1983
struct cudaMemLocation { 
#line 1984
cudaMemLocationType type; 
#line 1985
int id; 
#line 1986
}; 
#endif
#line 1991 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\driver_types.h"
#if 0
#line 1991
enum cudaMemAccessFlags { 
#line 1992
cudaMemAccessFlagsProtNone, 
#line 1993
cudaMemAccessFlagsProtRead, 
#line 1994
cudaMemAccessFlagsProtReadWrite = 3
#line 1995
}; 
#endif
#line 2000 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\driver_types.h"
#if 0
#line 2000
struct cudaMemAccessDesc { 
#line 2001
cudaMemLocation location; 
#line 2002
cudaMemAccessFlags flags; 
#line 2003
}; 
#endif
#line 2008 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\driver_types.h"
#if 0
#line 2008
enum cudaMemAllocationType { 
#line 2009
cudaMemAllocationTypeInvalid, 
#line 2013
cudaMemAllocationTypePinned, 
#line 2014
cudaMemAllocationTypeMax = 2147483647
#line 2015
}; 
#endif
#line 2020 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\driver_types.h"
#if 0
#line 2020
enum cudaMemAllocationHandleType { 
#line 2021
cudaMemHandleTypeNone, 
#line 2022
cudaMemHandleTypePosixFileDescriptor, 
#line 2023
cudaMemHandleTypeWin32, 
#line 2024
cudaMemHandleTypeWin32Kmt = 4
#line 2025
}; 
#endif
#line 2030 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\driver_types.h"
#if 0
#line 2030
struct cudaMemPoolProps { 
#line 2031
cudaMemAllocationType allocType; 
#line 2032
cudaMemAllocationHandleType handleTypes; 
#line 2033
cudaMemLocation location; 
#line 2040
void *win32SecurityAttributes; 
#line 2041
unsigned char reserved[64]; 
#line 2042
}; 
#endif
#line 2047 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\driver_types.h"
#if 0
#line 2047
struct cudaMemPoolPtrExportData { 
#line 2048
unsigned char reserved[64]; 
#line 2049
}; 
#endif
#line 2054 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\driver_types.h"
#if 0
#line 2054
struct cudaMemAllocNodeParams { 
#line 2059
cudaMemPoolProps poolProps; 
#line 2060
const cudaMemAccessDesc *accessDescs; 
#line 2061
size_t accessDescCount; 
#line 2062
size_t bytesize; 
#line 2063
void *dptr; 
#line 2064
}; 
#endif
#line 2069 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\driver_types.h"
#if 0
#line 2069
enum cudaGraphMemAttributeType { 
#line 2074
cudaGraphMemAttrUsedMemCurrent, 
#line 2081
cudaGraphMemAttrUsedMemHigh, 
#line 2088
cudaGraphMemAttrReservedMemCurrent, 
#line 2095
cudaGraphMemAttrReservedMemHigh
#line 2096
}; 
#endif
#line 2102 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\driver_types.h"
#if 0
#line 2102
enum cudaDeviceP2PAttr { 
#line 2103
cudaDevP2PAttrPerformanceRank = 1, 
#line 2104
cudaDevP2PAttrAccessSupported, 
#line 2105
cudaDevP2PAttrNativeAtomicSupported, 
#line 2106
cudaDevP2PAttrCudaArrayAccessSupported
#line 2107
}; 
#endif
#line 2114 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\driver_types.h"
#if 0
#line 2114
struct CUuuid_st { 
#line 2115
char bytes[16]; 
#line 2116
}; 
#endif
#line 2117 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\driver_types.h"
#if 0
typedef CUuuid_st 
#line 2117
CUuuid; 
#endif
#line 2119 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\driver_types.h"
#if 0
typedef CUuuid_st 
#line 2119
cudaUUID_t; 
#endif
#line 2124 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\driver_types.h"
#if 0
#line 2124
struct cudaDeviceProp { 
#line 2126
char name[256]; 
#line 2127
cudaUUID_t uuid; 
#line 2128
char luid[8]; 
#line 2129
unsigned luidDeviceNodeMask; 
#line 2130
size_t totalGlobalMem; 
#line 2131
size_t sharedMemPerBlock; 
#line 2132
int regsPerBlock; 
#line 2133
int warpSize; 
#line 2134
size_t memPitch; 
#line 2135
int maxThreadsPerBlock; 
#line 2136
int maxThreadsDim[3]; 
#line 2137
int maxGridSize[3]; 
#line 2138
int clockRate; 
#line 2139
size_t totalConstMem; 
#line 2140
int major; 
#line 2141
int minor; 
#line 2142
size_t textureAlignment; 
#line 2143
size_t texturePitchAlignment; 
#line 2144
int deviceOverlap; 
#line 2145
int multiProcessorCount; 
#line 2146
int kernelExecTimeoutEnabled; 
#line 2147
int integrated; 
#line 2148
int canMapHostMemory; 
#line 2149
int computeMode; 
#line 2150
int maxTexture1D; 
#line 2151
int maxTexture1DMipmap; 
#line 2152
int maxTexture1DLinear; 
#line 2153
int maxTexture2D[2]; 
#line 2154
int maxTexture2DMipmap[2]; 
#line 2155
int maxTexture2DLinear[3]; 
#line 2156
int maxTexture2DGather[2]; 
#line 2157
int maxTexture3D[3]; 
#line 2158
int maxTexture3DAlt[3]; 
#line 2159
int maxTextureCubemap; 
#line 2160
int maxTexture1DLayered[2]; 
#line 2161
int maxTexture2DLayered[3]; 
#line 2162
int maxTextureCubemapLayered[2]; 
#line 2163
int maxSurface1D; 
#line 2164
int maxSurface2D[2]; 
#line 2165
int maxSurface3D[3]; 
#line 2166
int maxSurface1DLayered[2]; 
#line 2167
int maxSurface2DLayered[3]; 
#line 2168
int maxSurfaceCubemap; 
#line 2169
int maxSurfaceCubemapLayered[2]; 
#line 2170
size_t surfaceAlignment; 
#line 2171
int concurrentKernels; 
#line 2172
int ECCEnabled; 
#line 2173
int pciBusID; 
#line 2174
int pciDeviceID; 
#line 2175
int pciDomainID; 
#line 2176
int tccDriver; 
#line 2177
int asyncEngineCount; 
#line 2178
int unifiedAddressing; 
#line 2179
int memoryClockRate; 
#line 2180
int memoryBusWidth; 
#line 2181
int l2CacheSize; 
#line 2182
int persistingL2CacheMaxSize; 
#line 2183
int maxThreadsPerMultiProcessor; 
#line 2184
int streamPrioritiesSupported; 
#line 2185
int globalL1CacheSupported; 
#line 2186
int localL1CacheSupported; 
#line 2187
size_t sharedMemPerMultiprocessor; 
#line 2188
int regsPerMultiprocessor; 
#line 2189
int managedMemory; 
#line 2190
int isMultiGpuBoard; 
#line 2191
int multiGpuBoardGroupID; 
#line 2192
int hostNativeAtomicSupported; 
#line 2193
int singleToDoublePrecisionPerfRatio; 
#line 2194
int pageableMemoryAccess; 
#line 2195
int concurrentManagedAccess; 
#line 2196
int computePreemptionSupported; 
#line 2197
int canUseHostPointerForRegisteredMem; 
#line 2198
int cooperativeLaunch; 
#line 2199
int cooperativeMultiDeviceLaunch; 
#line 2200
size_t sharedMemPerBlockOptin; 
#line 2201
int pageableMemoryAccessUsesHostPageTables; 
#line 2202
int directManagedMemAccessFromHost; 
#line 2203
int maxBlocksPerMultiProcessor; 
#line 2204
int accessPolicyMaxWindowSize; 
#line 2205
size_t reservedSharedMemPerBlock; 
#line 2206
}; 
#endif
#line 2302 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\driver_types.h"
#if 0
typedef 
#line 2299
struct cudaIpcEventHandle_st { 
#line 2301
char reserved[64]; 
#line 2302
} cudaIpcEventHandle_t; 
#endif
#line 2310 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\driver_types.h"
#if 0
typedef 
#line 2307
struct cudaIpcMemHandle_st { 
#line 2309
char reserved[64]; 
#line 2310
} cudaIpcMemHandle_t; 
#endif
#line 2315 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\driver_types.h"
#if 0
#line 2315
enum cudaExternalMemoryHandleType { 
#line 2319
cudaExternalMemoryHandleTypeOpaqueFd = 1, 
#line 2323
cudaExternalMemoryHandleTypeOpaqueWin32, 
#line 2327
cudaExternalMemoryHandleTypeOpaqueWin32Kmt, 
#line 2331
cudaExternalMemoryHandleTypeD3D12Heap, 
#line 2335
cudaExternalMemoryHandleTypeD3D12Resource, 
#line 2339
cudaExternalMemoryHandleTypeD3D11Resource, 
#line 2343
cudaExternalMemoryHandleTypeD3D11ResourceKmt, 
#line 2347
cudaExternalMemoryHandleTypeNvSciBuf
#line 2348
}; 
#endif
#line 2390 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\driver_types.h"
#if 0
#line 2390
struct cudaExternalMemoryHandleDesc { 
#line 2394
cudaExternalMemoryHandleType type; 
#line 2395
union { 
#line 2401
int fd; 
#line 2417
struct { 
#line 2421
void *handle; 
#line 2426
const void *name; 
#line 2427
} win32; 
#line 2432
const void *nvSciBufObject; 
#line 2433
} handle; 
#line 2437
unsigned __int64 size; 
#line 2441
unsigned flags; __pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)
#line 2442
}; 
#endif
#line 2447 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\driver_types.h"
#if 0
#line 2447
struct cudaExternalMemoryBufferDesc { 
#line 2451
unsigned __int64 offset; 
#line 2455
unsigned __int64 size; 
#line 2459
unsigned flags; 
#line 2460
}; 
#endif
#line 2465 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\driver_types.h"
#if 0
#line 2465
struct cudaExternalMemoryMipmappedArrayDesc { 
#line 2470
unsigned __int64 offset; 
#line 2474
cudaChannelFormatDesc formatDesc; 
#line 2478
cudaExtent extent; 
#line 2483
unsigned flags; 
#line 2487
unsigned numLevels; 
#line 2488
}; 
#endif
#line 2493 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\driver_types.h"
#if 0
#line 2493
enum cudaExternalSemaphoreHandleType { 
#line 2497
cudaExternalSemaphoreHandleTypeOpaqueFd = 1, 
#line 2501
cudaExternalSemaphoreHandleTypeOpaqueWin32, 
#line 2505
cudaExternalSemaphoreHandleTypeOpaqueWin32Kmt, 
#line 2509
cudaExternalSemaphoreHandleTypeD3D12Fence, 
#line 2513
cudaExternalSemaphoreHandleTypeD3D11Fence, 
#line 2517
cudaExternalSemaphoreHandleTypeNvSciSync, 
#line 2521
cudaExternalSemaphoreHandleTypeKeyedMutex, 
#line 2525
cudaExternalSemaphoreHandleTypeKeyedMutexKmt, 
#line 2529
cudaExternalSemaphoreHandleTypeTimelineSemaphoreFd, 
#line 2533
cudaExternalSemaphoreHandleTypeTimelineSemaphoreWin32
#line 2534
}; 
#endif
#line 2539 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\driver_types.h"
#if 0
#line 2539
struct cudaExternalSemaphoreHandleDesc { 
#line 2543
cudaExternalSemaphoreHandleType type; 
#line 2544
union { 
#line 2551
int fd; 
#line 2567
struct { 
#line 2571
void *handle; 
#line 2576
const void *name; 
#line 2577
} win32; 
#line 2581
const void *nvSciSyncObj; 
#line 2582
} handle; 
#line 2586
unsigned flags; __pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)
#line 2587
}; 
#endif
#line 2592 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\driver_types.h"
#if 0
#line 2592
struct cudaExternalSemaphoreSignalParams_v1 { 
#line 2593
struct { 
#line 2597
struct { 
#line 2601
unsigned __int64 value; 
#line 2602
} fence; 
#line 2603
union { 
#line 2608
void *fence; 
#line 2609
unsigned __int64 reserved; 
#line 2610
} nvSciSync; 
#line 2614
struct { 
#line 2618
unsigned __int64 key; 
#line 2619
} keyedMutex; 
#line 2620
} params; 
#line 2631
unsigned flags; 
#line 2632
}; 
#endif
#line 2637 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\driver_types.h"
#if 0
#line 2637
struct cudaExternalSemaphoreWaitParams_v1 { 
#line 2638
struct { 
#line 2642
struct { 
#line 2646
unsigned __int64 value; 
#line 2647
} fence; 
#line 2648
union { 
#line 2653
void *fence; 
#line 2654
unsigned __int64 reserved; 
#line 2655
} nvSciSync; 
#line 2659
struct { 
#line 2663
unsigned __int64 key; 
#line 2667
unsigned timeoutMs; 
#line 2668
} keyedMutex; 
#line 2669
} params; 
#line 2680
unsigned flags; 
#line 2681
}; 
#endif
#line 2686 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\driver_types.h"
#if 0
#line 2686
struct cudaExternalSemaphoreSignalParams { 
#line 2687
struct { 
#line 2691
struct { 
#line 2695
unsigned __int64 value; 
#line 2696
} fence; 
#line 2697
union { 
#line 2702
void *fence; 
#line 2703
unsigned __int64 reserved; 
#line 2704
} nvSciSync; 
#line 2708
struct { 
#line 2712
unsigned __int64 key; 
#line 2713
} keyedMutex; 
#line 2714
unsigned reserved[12]; 
#line 2715
} params; 
#line 2726
unsigned flags; 
#line 2727
unsigned reserved[16]; 
#line 2728
}; 
#endif
#line 2733 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\driver_types.h"
#if 0
#line 2733
struct cudaExternalSemaphoreWaitParams { 
#line 2734
struct { 
#line 2738
struct { 
#line 2742
unsigned __int64 value; 
#line 2743
} fence; 
#line 2744
union { 
#line 2749
void *fence; 
#line 2750
unsigned __int64 reserved; 
#line 2751
} nvSciSync; 
#line 2755
struct { 
#line 2759
unsigned __int64 key; 
#line 2763
unsigned timeoutMs; 
#line 2764
} keyedMutex; 
#line 2765
unsigned reserved[10]; 
#line 2766
} params; 
#line 2777
unsigned flags; 
#line 2778
unsigned reserved[16]; 
#line 2779
}; 
#endif
#line 2790 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\driver_types.h"
#if 0
typedef cudaError 
#line 2790
cudaError_t; 
#endif
#line 2795 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\driver_types.h"
#if 0
typedef struct CUstream_st *
#line 2795
cudaStream_t; 
#endif
#line 2800 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\driver_types.h"
#if 0
typedef struct CUevent_st *
#line 2800
cudaEvent_t; 
#endif
#line 2805 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\driver_types.h"
#if 0
typedef cudaGraphicsResource *
#line 2805
cudaGraphicsResource_t; 
#endif
#line 2810 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\driver_types.h"
#if 0
typedef cudaOutputMode 
#line 2810
cudaOutputMode_t; 
#endif
#line 2815 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\driver_types.h"
#if 0
typedef struct CUexternalMemory_st *
#line 2815
cudaExternalMemory_t; 
#endif
#line 2820 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\driver_types.h"
#if 0
typedef struct CUexternalSemaphore_st *
#line 2820
cudaExternalSemaphore_t; 
#endif
#line 2825 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\driver_types.h"
#if 0
typedef struct CUgraph_st *
#line 2825
cudaGraph_t; 
#endif
#line 2830 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\driver_types.h"
#if 0
typedef struct CUgraphNode_st *
#line 2830
cudaGraphNode_t; 
#endif
#line 2835 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\driver_types.h"
#if 0
typedef struct CUuserObject_st *
#line 2835
cudaUserObject_t; 
#endif
#line 2840 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\driver_types.h"
#if 0
typedef struct CUfunc_st *
#line 2840
cudaFunction_t; 
#endif
#line 2845 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\driver_types.h"
#if 0
typedef struct CUmemPoolHandle_st *
#line 2845
cudaMemPool_t; 
#endif
#line 2850 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\driver_types.h"
#if 0
#line 2850
enum cudaCGScope { 
#line 2851
cudaCGScopeInvalid, 
#line 2852
cudaCGScopeGrid, 
#line 2853
cudaCGScopeMultiGrid
#line 2854
}; 
#endif
#line 2859 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\driver_types.h"
#if 0
#line 2859
struct cudaLaunchParams { 
#line 2861
void *func; 
#line 2862
dim3 gridDim; 
#line 2863
dim3 blockDim; 
#line 2864
void **args; 
#line 2865
size_t sharedMem; 
#line 2866
cudaStream_t stream; 
#line 2867
}; 
#endif
#line 2872 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\driver_types.h"
#if 0
#line 2872
struct cudaKernelNodeParams { 
#line 2873
void *func; 
#line 2874
dim3 gridDim; 
#line 2875
dim3 blockDim; 
#line 2876
unsigned sharedMemBytes; 
#line 2877
void **kernelParams; 
#line 2878
void **extra; 
#line 2879
}; 
#endif
#line 2884 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\driver_types.h"
#if 0
#line 2884
struct cudaExternalSemaphoreSignalNodeParams { 
#line 2885
cudaExternalSemaphore_t *extSemArray; 
#line 2886
const cudaExternalSemaphoreSignalParams *paramsArray; 
#line 2887
unsigned numExtSems; 
#line 2888
}; 
#endif
#line 2893 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\driver_types.h"
#if 0
#line 2893
struct cudaExternalSemaphoreWaitNodeParams { 
#line 2894
cudaExternalSemaphore_t *extSemArray; 
#line 2895
const cudaExternalSemaphoreWaitParams *paramsArray; 
#line 2896
unsigned numExtSems; 
#line 2897
}; 
#endif
#line 2902 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\driver_types.h"
#if 0
#line 2902
enum cudaGraphNodeType { 
#line 2903
cudaGraphNodeTypeKernel, 
#line 2904
cudaGraphNodeTypeMemcpy, 
#line 2905
cudaGraphNodeTypeMemset, 
#line 2906
cudaGraphNodeTypeHost, 
#line 2907
cudaGraphNodeTypeGraph, 
#line 2908
cudaGraphNodeTypeEmpty, 
#line 2909
cudaGraphNodeTypeWaitEvent, 
#line 2910
cudaGraphNodeTypeEventRecord, 
#line 2911
cudaGraphNodeTypeExtSemaphoreSignal, 
#line 2912
cudaGraphNodeTypeExtSemaphoreWait, 
#line 2913
cudaGraphNodeTypeMemAlloc, 
#line 2914
cudaGraphNodeTypeMemFree, 
#line 2915
cudaGraphNodeTypeCount
#line 2916
}; 
#endif
#line 2921 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\driver_types.h"
typedef struct CUgraphExec_st *cudaGraphExec_t; 
#line 2926
#if 0
#line 2926
enum cudaGraphExecUpdateResult { 
#line 2927
cudaGraphExecUpdateSuccess, 
#line 2928
cudaGraphExecUpdateError, 
#line 2929
cudaGraphExecUpdateErrorTopologyChanged, 
#line 2930
cudaGraphExecUpdateErrorNodeTypeChanged, 
#line 2931
cudaGraphExecUpdateErrorFunctionChanged, 
#line 2932
cudaGraphExecUpdateErrorParametersChanged, 
#line 2933
cudaGraphExecUpdateErrorNotSupported, 
#line 2934
cudaGraphExecUpdateErrorUnsupportedFunctionChange, 
#line 2935
cudaGraphExecUpdateErrorAttributesChanged
#line 2936
}; 
#endif
#line 2942 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\driver_types.h"
#if 0
#line 2942
enum cudaGetDriverEntryPointFlags { 
#line 2943
cudaEnableDefault, 
#line 2944
cudaEnableLegacyStream, 
#line 2945
cudaEnablePerThreadDefaultStream
#line 2946
}; 
#endif
#line 2951 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\driver_types.h"
#if 0
#line 2951
enum cudaGraphDebugDotFlags { 
#line 2952
cudaGraphDebugDotFlagsVerbose = (1 << 0), 
#line 2953
cudaGraphDebugDotFlagsKernelNodeParams = (1 << 2), 
#line 2954
cudaGraphDebugDotFlagsMemcpyNodeParams = (1 << 3), 
#line 2955
cudaGraphDebugDotFlagsMemsetNodeParams = (1 << 4), 
#line 2956
cudaGraphDebugDotFlagsHostNodeParams = (1 << 5), 
#line 2957
cudaGraphDebugDotFlagsEventNodeParams = (1 << 6), 
#line 2958
cudaGraphDebugDotFlagsExtSemasSignalNodeParams = (1 << 7), 
#line 2959
cudaGraphDebugDotFlagsExtSemasWaitNodeParams = (1 << 8), 
#line 2960
cudaGraphDebugDotFlagsKernelNodeAttributes = (1 << 9), 
#line 2961
cudaGraphDebugDotFlagsHandles = (1 << 10)
#line 2962
}; 
#endif
#line 2967 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\driver_types.h"
#if 0
#line 2967
enum cudaGraphInstantiateFlags { 
#line 2968
cudaGraphInstantiateFlagAutoFreeOnLaunch = 1, 
#line 2969
cudaGraphInstantiateFlagUseNodePriority = 8
#line 2971
}; 
#endif
#line 3010 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\driver_types.h"
#if 0
typedef 
#line 2976
enum cudaLaunchAttributeID { 
#line 2977
cudaLaunchAttributeIgnore, 
#line 2978
cudaLaunchAttributeAccessPolicyWindow, 
#line 2979
cudaLaunchAttributeCooperative, 
#line 2980
cudaLaunchAttributeSynchronizationPolicy, 
#line 2981
cudaLaunchAttributeClusterDimension, 
#line 2982
cudaLaunchAttributeClusterSchedulingPolicyPreference, 
#line 2983
cudaLaunchAttributeProgrammaticStreamSerialization, 
#line 2991
cudaLaunchAttributeProgrammaticEvent, 
#line 3009
cudaLaunchAttributePriority
#line 3010
} cudaLaunchAttributeID; 
#endif
#line 3033 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\driver_types.h"
#if 0
typedef 
#line 3015
union cudaLaunchAttributeValue { 
#line 3016
char pad[64]; 
#line 3017
cudaAccessPolicyWindow accessPolicyWindow; 
#line 3018
int cooperative; 
#line 3019
cudaSynchronizationPolicy syncPolicy; 
#line 3020
struct { 
#line 3021
unsigned x; 
#line 3022
unsigned y; 
#line 3023
unsigned z; 
#line 3024
} clusterDim; 
#line 3025
cudaClusterSchedulingPolicy clusterSchedulingPolicyPreference; 
#line 3026
int programmaticStreamSerializationAllowed; 
#line 3027
struct { 
#line 3028
cudaEvent_t event; 
#line 3029
int flags; 
#line 3030
int triggerAtBlockStart; 
#line 3031
} programmaticEvent; 
#line 3032
int priority; __pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)
#line 3033
} cudaLaunchAttributeValue; 
#endif
#line 3042 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\driver_types.h"
#if 0
typedef 
#line 3038
struct cudaLaunchAttribute_st { 
#line 3039
cudaLaunchAttributeID id; 
#line 3040
char pad[(8) - sizeof(cudaLaunchAttributeID)]; 
#line 3041
cudaLaunchAttributeValue val; 
#line 3042
} cudaLaunchAttribute; 
#endif
#line 3054 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\driver_types.h"
#if 0
typedef 
#line 3047
struct cudaLaunchConfig_st { 
#line 3048
dim3 gridDim; 
#line 3049
dim3 blockDim; 
#line 3050
size_t dynamicSmemBytes; 
#line 3051
cudaStream_t stream; 
#line 3052
cudaLaunchAttribute *attrs; 
#line 3053
unsigned numAttrs; __pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)
#line 3054
} cudaLaunchConfig_t; 
#endif
#line 84 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\surface_types.h"
#if 0
#line 84
enum cudaSurfaceBoundaryMode { 
#line 86
cudaBoundaryModeZero, 
#line 87
cudaBoundaryModeClamp, 
#line 88
cudaBoundaryModeTrap
#line 89
}; 
#endif
#line 94 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\surface_types.h"
#if 0
#line 94
enum cudaSurfaceFormatMode { 
#line 96
cudaFormatModeForced, 
#line 97
cudaFormatModeAuto
#line 98
}; 
#endif
#line 103 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\surface_types.h"
#if 0
#line 103
struct surfaceReference { 
#line 108
cudaChannelFormatDesc channelDesc; 
#line 109
}; 
#endif
#line 114 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\surface_types.h"
#if 0
typedef unsigned __int64 
#line 114
cudaSurfaceObject_t; 
#endif
#line 84 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\texture_types.h"
#if 0
#line 84
enum cudaTextureAddressMode { 
#line 86
cudaAddressModeWrap, 
#line 87
cudaAddressModeClamp, 
#line 88
cudaAddressModeMirror, 
#line 89
cudaAddressModeBorder
#line 90
}; 
#endif
#line 95 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\texture_types.h"
#if 0
#line 95
enum cudaTextureFilterMode { 
#line 97
cudaFilterModePoint, 
#line 98
cudaFilterModeLinear
#line 99
}; 
#endif
#line 104 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\texture_types.h"
#if 0
#line 104
enum cudaTextureReadMode { 
#line 106
cudaReadModeElementType, 
#line 107
cudaReadModeNormalizedFloat
#line 108
}; 
#endif
#line 113 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\texture_types.h"
#if 0
#line 113
struct textureReference { 
#line 118
int normalized; 
#line 122
cudaTextureFilterMode filterMode; 
#line 126
cudaTextureAddressMode addressMode[3]; 
#line 130
cudaChannelFormatDesc channelDesc; 
#line 134
int sRGB; 
#line 138
unsigned maxAnisotropy; 
#line 142
cudaTextureFilterMode mipmapFilterMode; 
#line 146
float mipmapLevelBias; 
#line 150
float minMipmapLevelClamp; 
#line 154
float maxMipmapLevelClamp; 
#line 158
int disableTrilinearOptimization; 
#line 159
int __cudaReserved[14]; 
#line 160
}; 
#endif
#line 165 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\texture_types.h"
#if 0
#line 165
struct cudaTextureDesc { 
#line 170
cudaTextureAddressMode addressMode[3]; 
#line 174
cudaTextureFilterMode filterMode; 
#line 178
cudaTextureReadMode readMode; 
#line 182
int sRGB; 
#line 186
float borderColor[4]; 
#line 190
int normalizedCoords; 
#line 194
unsigned maxAnisotropy; 
#line 198
cudaTextureFilterMode mipmapFilterMode; 
#line 202
float mipmapLevelBias; 
#line 206
float minMipmapLevelClamp; 
#line 210
float maxMipmapLevelClamp; 
#line 214
int disableTrilinearOptimization; 
#line 215
}; 
#endif
#line 217 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\texture_types.h"
#if 0
#line 217
struct cudaTextureDesc_v2 { 
#line 222
cudaTextureAddressMode addressMode[3]; 
#line 226
cudaTextureFilterMode filterMode; 
#line 230
cudaTextureReadMode readMode; 
#line 234
int sRGB; 
#line 238
float borderColor[4]; 
#line 242
int normalizedCoords; 
#line 246
unsigned maxAnisotropy; 
#line 250
cudaTextureFilterMode mipmapFilterMode; 
#line 254
float mipmapLevelBias; 
#line 258
float minMipmapLevelClamp; 
#line 262
float maxMipmapLevelClamp; 
#line 266
int disableTrilinearOptimization; 
#line 270
int seamlessCubemap; 
#line 271
}; 
#endif
#line 276 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\texture_types.h"
#if 0
typedef unsigned __int64 
#line 276
cudaTextureObject_t; 
#endif
#line 87 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\library_types.h"
typedef 
#line 55
enum cudaDataType_t { 
#line 57
CUDA_R_16F = 2, 
#line 58
CUDA_C_16F = 6, 
#line 59
CUDA_R_16BF = 14, 
#line 60
CUDA_C_16BF, 
#line 61
CUDA_R_32F = 0, 
#line 62
CUDA_C_32F = 4, 
#line 63
CUDA_R_64F = 1, 
#line 64
CUDA_C_64F = 5, 
#line 65
CUDA_R_4I = 16, 
#line 66
CUDA_C_4I, 
#line 67
CUDA_R_4U, 
#line 68
CUDA_C_4U, 
#line 69
CUDA_R_8I = 3, 
#line 70
CUDA_C_8I = 7, 
#line 71
CUDA_R_8U, 
#line 72
CUDA_C_8U, 
#line 73
CUDA_R_16I = 20, 
#line 74
CUDA_C_16I, 
#line 75
CUDA_R_16U, 
#line 76
CUDA_C_16U, 
#line 77
CUDA_R_32I = 10, 
#line 78
CUDA_C_32I, 
#line 79
CUDA_R_32U, 
#line 80
CUDA_C_32U, 
#line 81
CUDA_R_64I = 24, 
#line 82
CUDA_C_64I, 
#line 83
CUDA_R_64U, 
#line 84
CUDA_C_64U, 
#line 85
CUDA_R_8F_E4M3, 
#line 86
CUDA_R_8F_E5M2
#line 87
} cudaDataType; 
#line 95
typedef 
#line 90
enum libraryPropertyType_t { 
#line 92
MAJOR_VERSION, 
#line 93
MINOR_VERSION, 
#line 94
PATCH_LEVEL
#line 95
} libraryPropertyType; 
#line 131 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\cuda_device_runtime_api.h"
extern "C" {
#line 133
extern cudaError_t __stdcall cudaDeviceGetAttribute(int * value, cudaDeviceAttr attr, int device); 
#line 134
extern cudaError_t __stdcall cudaDeviceGetLimit(size_t * pValue, cudaLimit limit); 
#line 135
extern cudaError_t __stdcall cudaDeviceGetCacheConfig(cudaFuncCache * pCacheConfig); 
#line 136
extern cudaError_t __stdcall cudaDeviceGetSharedMemConfig(cudaSharedMemConfig * pConfig); 
#line 139
extern cudaError_t __stdcall cudaDeviceSynchronize(); 
#line 141 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\cuda_device_runtime_api.h"
extern cudaError_t __stdcall __cudaDeviceSynchronizeDeprecationAvoidance(); 
#line 142
extern cudaError_t __stdcall cudaGetLastError(); 
#line 143
extern cudaError_t __stdcall cudaPeekAtLastError(); 
#line 144
extern const char *__stdcall cudaGetErrorString(cudaError_t error); 
#line 145
extern const char *__stdcall cudaGetErrorName(cudaError_t error); 
#line 146
extern cudaError_t __stdcall cudaGetDeviceCount(int * count); 
#line 147
extern cudaError_t __stdcall cudaGetDevice(int * device); 
#line 148
extern cudaError_t __stdcall cudaStreamCreateWithFlags(cudaStream_t * pStream, unsigned flags); 
#line 149
extern cudaError_t __stdcall cudaStreamDestroy(cudaStream_t stream); 
#line 150
extern cudaError_t __stdcall cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned flags); 
#line 151
extern cudaError_t __stdcall cudaStreamWaitEvent_ptsz(cudaStream_t stream, cudaEvent_t event, unsigned flags); 
#line 152
extern cudaError_t __stdcall cudaEventCreateWithFlags(cudaEvent_t * event, unsigned flags); 
#line 153
extern cudaError_t __stdcall cudaEventRecord(cudaEvent_t event, cudaStream_t stream); 
#line 154
extern cudaError_t __stdcall cudaEventRecord_ptsz(cudaEvent_t event, cudaStream_t stream); 
#line 155
extern cudaError_t __stdcall cudaEventRecordWithFlags(cudaEvent_t event, cudaStream_t stream, unsigned flags); 
#line 156
extern cudaError_t __stdcall cudaEventRecordWithFlags_ptsz(cudaEvent_t event, cudaStream_t stream, unsigned flags); 
#line 157
extern cudaError_t __stdcall cudaEventDestroy(cudaEvent_t event); 
#line 158
extern cudaError_t __stdcall cudaFuncGetAttributes(cudaFuncAttributes * attr, const void * func); 
#line 159
extern cudaError_t __stdcall cudaFree(void * devPtr); 
#line 160
extern cudaError_t __stdcall cudaMalloc(void ** devPtr, size_t size); 
#line 161
extern cudaError_t __stdcall cudaMemcpyAsync(void * dst, const void * src, size_t count, cudaMemcpyKind kind, cudaStream_t stream); 
#line 162
extern cudaError_t __stdcall cudaMemcpyAsync_ptsz(void * dst, const void * src, size_t count, cudaMemcpyKind kind, cudaStream_t stream); 
#line 163
extern cudaError_t __stdcall cudaMemcpy2DAsync(void * dst, size_t dpitch, const void * src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream); 
#line 164
extern cudaError_t __stdcall cudaMemcpy2DAsync_ptsz(void * dst, size_t dpitch, const void * src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream); 
#line 165
extern cudaError_t __stdcall cudaMemcpy3DAsync(const cudaMemcpy3DParms * p, cudaStream_t stream); 
#line 166
extern cudaError_t __stdcall cudaMemcpy3DAsync_ptsz(const cudaMemcpy3DParms * p, cudaStream_t stream); 
#line 167
extern cudaError_t __stdcall cudaMemsetAsync(void * devPtr, int value, size_t count, cudaStream_t stream); 
#line 168
extern cudaError_t __stdcall cudaMemsetAsync_ptsz(void * devPtr, int value, size_t count, cudaStream_t stream); 
#line 169
extern cudaError_t __stdcall cudaMemset2DAsync(void * devPtr, size_t pitch, int value, size_t width, size_t height, cudaStream_t stream); 
#line 170
extern cudaError_t __stdcall cudaMemset2DAsync_ptsz(void * devPtr, size_t pitch, int value, size_t width, size_t height, cudaStream_t stream); 
#line 171
extern cudaError_t __stdcall cudaMemset3DAsync(cudaPitchedPtr pitchedDevPtr, int value, cudaExtent extent, cudaStream_t stream); 
#line 172
extern cudaError_t __stdcall cudaMemset3DAsync_ptsz(cudaPitchedPtr pitchedDevPtr, int value, cudaExtent extent, cudaStream_t stream); 
#line 173
extern cudaError_t __stdcall cudaRuntimeGetVersion(int * runtimeVersion); 
#line 194
extern void *__stdcall cudaGetParameterBuffer(size_t alignment, size_t size); 
#line 222
extern void *__stdcall cudaGetParameterBufferV2(void * func, dim3 gridDimension, dim3 blockDimension, unsigned sharedMemSize); 
#line 223
extern cudaError_t __stdcall cudaLaunchDevice_ptsz(void * func, void * parameterBuffer, dim3 gridDimension, dim3 blockDimension, unsigned sharedMemSize, cudaStream_t stream); 
#line 224
extern cudaError_t __stdcall cudaLaunchDeviceV2_ptsz(void * parameterBuffer, cudaStream_t stream); 
#line 242 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\cuda_device_runtime_api.h"
extern cudaError_t __stdcall cudaLaunchDevice(void * func, void * parameterBuffer, dim3 gridDimension, dim3 blockDimension, unsigned sharedMemSize, cudaStream_t stream); 
#line 243
extern cudaError_t __stdcall cudaLaunchDeviceV2(void * parameterBuffer, cudaStream_t stream); 
#line 246 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\cuda_device_runtime_api.h"
extern cudaError_t __stdcall cudaOccupancyMaxActiveBlocksPerMultiprocessor(int * numBlocks, const void * func, int blockSize, size_t dynamicSmemSize); 
#line 247
extern cudaError_t __stdcall cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int * numBlocks, const void * func, int blockSize, size_t dynamicSmemSize, unsigned flags); 
#line 249
extern unsigned __int64 __stdcall cudaCGGetIntrinsicHandle(cudaCGScope scope); 
#line 250
extern cudaError_t __stdcall cudaCGSynchronize(unsigned __int64 handle, unsigned flags); 
#line 251
extern cudaError_t __stdcall cudaCGSynchronizeGrid(unsigned __int64 handle, unsigned flags); 
#line 252
extern cudaError_t __stdcall cudaCGGetSize(unsigned * numThreads, unsigned * numGrids, unsigned __int64 handle); 
#line 253
extern cudaError_t __stdcall cudaCGGetRank(unsigned * threadRank, unsigned * gridRank, unsigned __int64 handle); 
#line 254
}
#line 256
template< class T> static __inline cudaError_t cudaMalloc(T ** devPtr, size_t size); 
#line 257
template< class T> static __inline cudaError_t cudaFuncGetAttributes(cudaFuncAttributes * attr, T * entry); 
#line 258
template< class T> static __inline cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int * numBlocks, T func, int blockSize, size_t dynamicSmemSize); 
#line 259
template< class T> static __inline cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int * numBlocks, T func, int blockSize, size_t dynamicSmemSize, unsigned flags); 
#line 267 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\cuda_runtime_api.h"
extern "C" {
#line 307 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaDeviceReset(); 
#line 329
extern cudaError_t __stdcall cudaDeviceSynchronize(); 
#line 416
extern cudaError_t __stdcall cudaDeviceSetLimit(cudaLimit limit, size_t value); 
#line 449
extern cudaError_t __stdcall cudaDeviceGetLimit(size_t * pValue, cudaLimit limit); 
#line 472
extern cudaError_t __stdcall cudaDeviceGetTexture1DLinearMaxWidth(size_t * maxWidthInElements, const cudaChannelFormatDesc * fmtDesc, int device); 
#line 506 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaDeviceGetCacheConfig(cudaFuncCache * pCacheConfig); 
#line 543
extern cudaError_t __stdcall cudaDeviceGetStreamPriorityRange(int * leastPriority, int * greatestPriority); 
#line 587
extern cudaError_t __stdcall cudaDeviceSetCacheConfig(cudaFuncCache cacheConfig); 
#line 618
extern cudaError_t __stdcall cudaDeviceGetSharedMemConfig(cudaSharedMemConfig * pConfig); 
#line 662
extern cudaError_t __stdcall cudaDeviceSetSharedMemConfig(cudaSharedMemConfig config); 
#line 689
extern cudaError_t __stdcall cudaDeviceGetByPCIBusId(int * device, const char * pciBusId); 
#line 719
extern cudaError_t __stdcall cudaDeviceGetPCIBusId(char * pciBusId, int len, int device); 
#line 767
extern cudaError_t __stdcall cudaIpcGetEventHandle(cudaIpcEventHandle_t * handle, cudaEvent_t event); 
#line 808
extern cudaError_t __stdcall cudaIpcOpenEventHandle(cudaEvent_t * event, cudaIpcEventHandle_t handle); 
#line 851
extern cudaError_t __stdcall cudaIpcGetMemHandle(cudaIpcMemHandle_t * handle, void * devPtr); 
#line 915
extern cudaError_t __stdcall cudaIpcOpenMemHandle(void ** devPtr, cudaIpcMemHandle_t handle, unsigned flags); 
#line 951
extern cudaError_t __stdcall cudaIpcCloseMemHandle(void * devPtr); 
#line 983
extern cudaError_t __stdcall cudaDeviceFlushGPUDirectRDMAWrites(cudaFlushGPUDirectRDMAWritesTarget target, cudaFlushGPUDirectRDMAWritesScope scope); 
#line 1026 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\cuda_runtime_api.h"
__declspec(deprecated) extern cudaError_t __stdcall cudaThreadExit(); 
#line 1052
__declspec(deprecated) extern cudaError_t __stdcall cudaThreadSynchronize(); 
#line 1101
__declspec(deprecated) extern cudaError_t __stdcall cudaThreadSetLimit(cudaLimit limit, size_t value); 
#line 1134
__declspec(deprecated) extern cudaError_t __stdcall cudaThreadGetLimit(size_t * pValue, cudaLimit limit); 
#line 1170
__declspec(deprecated) extern cudaError_t __stdcall cudaThreadGetCacheConfig(cudaFuncCache * pCacheConfig); 
#line 1217
__declspec(deprecated) extern cudaError_t __stdcall cudaThreadSetCacheConfig(cudaFuncCache cacheConfig); 
#line 1278
extern cudaError_t __stdcall cudaGetLastError(); 
#line 1326
extern cudaError_t __stdcall cudaPeekAtLastError(); 
#line 1342
extern const char *__stdcall cudaGetErrorName(cudaError_t error); 
#line 1358
extern const char *__stdcall cudaGetErrorString(cudaError_t error); 
#line 1386
extern cudaError_t __stdcall cudaGetDeviceCount(int * count); 
#line 1659
extern cudaError_t __stdcall cudaGetDeviceProperties(cudaDeviceProp * prop, int device); 
#line 1859
extern cudaError_t __stdcall cudaDeviceGetAttribute(int * value, cudaDeviceAttr attr, int device); 
#line 1877
extern cudaError_t __stdcall cudaDeviceGetDefaultMemPool(cudaMemPool_t * memPool, int device); 
#line 1901
extern cudaError_t __stdcall cudaDeviceSetMemPool(int device, cudaMemPool_t memPool); 
#line 1921
extern cudaError_t __stdcall cudaDeviceGetMemPool(cudaMemPool_t * memPool, int device); 
#line 1969
extern cudaError_t __stdcall cudaDeviceGetNvSciSyncAttributes(void * nvSciSyncAttrList, int device, int flags); 
#line 2009
extern cudaError_t __stdcall cudaDeviceGetP2PAttribute(int * value, cudaDeviceP2PAttr attr, int srcDevice, int dstDevice); 
#line 2030
extern cudaError_t __stdcall cudaChooseDevice(int * device, const cudaDeviceProp * prop); 
#line 2074
extern cudaError_t __stdcall cudaSetDevice(int device); 
#line 2095
extern cudaError_t __stdcall cudaGetDevice(int * device); 
#line 2126
extern cudaError_t __stdcall cudaSetValidDevices(int * device_arr, int len); 
#line 2191
extern cudaError_t __stdcall cudaSetDeviceFlags(unsigned flags); 
#line 2235
extern cudaError_t __stdcall cudaGetDeviceFlags(unsigned * flags); 
#line 2275
extern cudaError_t __stdcall cudaStreamCreate(cudaStream_t * pStream); 
#line 2307
extern cudaError_t __stdcall cudaStreamCreateWithFlags(cudaStream_t * pStream, unsigned flags); 
#line 2353
extern cudaError_t __stdcall cudaStreamCreateWithPriority(cudaStream_t * pStream, unsigned flags, int priority); 
#line 2380
extern cudaError_t __stdcall cudaStreamGetPriority(cudaStream_t hStream, int * priority); 
#line 2405
extern cudaError_t __stdcall cudaStreamGetFlags(cudaStream_t hStream, unsigned * flags); 
#line 2420
extern cudaError_t __stdcall cudaCtxResetPersistingL2Cache(); 
#line 2440
extern cudaError_t __stdcall cudaStreamCopyAttributes(cudaStream_t dst, cudaStream_t src); 
#line 2461
extern cudaError_t __stdcall cudaStreamGetAttribute(cudaStream_t hStream, cudaLaunchAttributeID attr, cudaLaunchAttributeValue * value_out); 
#line 2485
extern cudaError_t __stdcall cudaStreamSetAttribute(cudaStream_t hStream, cudaLaunchAttributeID attr, const cudaLaunchAttributeValue * value); 
#line 2519
extern cudaError_t __stdcall cudaStreamDestroy(cudaStream_t stream); 
#line 2550
extern cudaError_t __stdcall cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned flags = 0); 
#line 2558
typedef void (__stdcall *cudaStreamCallback_t)(cudaStream_t stream, cudaError_t status, void * userData); 
#line 2625
extern cudaError_t __stdcall cudaStreamAddCallback(cudaStream_t stream, cudaStreamCallback_t callback, void * userData, unsigned flags); 
#line 2649
extern cudaError_t __stdcall cudaStreamSynchronize(cudaStream_t stream); 
#line 2674
extern cudaError_t __stdcall cudaStreamQuery(cudaStream_t stream); 
#line 2758
extern cudaError_t __stdcall cudaStreamAttachMemAsync(cudaStream_t stream, void * devPtr, size_t length = 0, unsigned flags = 4); 
#line 2797 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaStreamBeginCapture(cudaStream_t stream, cudaStreamCaptureMode mode); 
#line 2848
extern cudaError_t __stdcall cudaThreadExchangeStreamCaptureMode(cudaStreamCaptureMode * mode); 
#line 2876
extern cudaError_t __stdcall cudaStreamEndCapture(cudaStream_t stream, cudaGraph_t * pGraph); 
#line 2914
extern cudaError_t __stdcall cudaStreamIsCapturing(cudaStream_t stream, cudaStreamCaptureStatus * pCaptureStatus); 
#line 2946
extern cudaError_t __stdcall cudaStreamGetCaptureInfo(cudaStream_t stream, cudaStreamCaptureStatus * pCaptureStatus, unsigned __int64 * pId); 
#line 3001
extern cudaError_t __stdcall cudaStreamGetCaptureInfo_v2(cudaStream_t stream, cudaStreamCaptureStatus * captureStatus_out, unsigned __int64 * id_out = 0, cudaGraph_t * graph_out = 0, const cudaGraphNode_t ** dependencies_out = 0, size_t * numDependencies_out = 0); 
#line 3034
extern cudaError_t __stdcall cudaStreamUpdateCaptureDependencies(cudaStream_t stream, cudaGraphNode_t * dependencies, size_t numDependencies, unsigned flags = 0); 
#line 3071
extern cudaError_t __stdcall cudaEventCreate(cudaEvent_t * event); 
#line 3108
extern cudaError_t __stdcall cudaEventCreateWithFlags(cudaEvent_t * event, unsigned flags); 
#line 3148
extern cudaError_t __stdcall cudaEventRecord(cudaEvent_t event, cudaStream_t stream = 0); 
#line 3195
extern cudaError_t __stdcall cudaEventRecordWithFlags(cudaEvent_t event, cudaStream_t stream = 0, unsigned flags = 0); 
#line 3227 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaEventQuery(cudaEvent_t event); 
#line 3257
extern cudaError_t __stdcall cudaEventSynchronize(cudaEvent_t event); 
#line 3286
extern cudaError_t __stdcall cudaEventDestroy(cudaEvent_t event); 
#line 3330
extern cudaError_t __stdcall cudaEventElapsedTime(float * ms, cudaEvent_t start, cudaEvent_t end); 
#line 3510
extern cudaError_t __stdcall cudaImportExternalMemory(cudaExternalMemory_t * extMem_out, const cudaExternalMemoryHandleDesc * memHandleDesc); 
#line 3565
extern cudaError_t __stdcall cudaExternalMemoryGetMappedBuffer(void ** devPtr, cudaExternalMemory_t extMem, const cudaExternalMemoryBufferDesc * bufferDesc); 
#line 3627
extern cudaError_t __stdcall cudaExternalMemoryGetMappedMipmappedArray(cudaMipmappedArray_t * mipmap, cudaExternalMemory_t extMem, const cudaExternalMemoryMipmappedArrayDesc * mipmapDesc); 
#line 3651
extern cudaError_t __stdcall cudaDestroyExternalMemory(cudaExternalMemory_t extMem); 
#line 3804
extern cudaError_t __stdcall cudaImportExternalSemaphore(cudaExternalSemaphore_t * extSem_out, const cudaExternalSemaphoreHandleDesc * semHandleDesc); 
#line 3871
extern cudaError_t __stdcall cudaSignalExternalSemaphoresAsync_v2(const cudaExternalSemaphore_t * extSemArray, const cudaExternalSemaphoreSignalParams * paramsArray, unsigned numExtSems, cudaStream_t stream = 0); 
#line 3947
extern cudaError_t __stdcall cudaWaitExternalSemaphoresAsync_v2(const cudaExternalSemaphore_t * extSemArray, const cudaExternalSemaphoreWaitParams * paramsArray, unsigned numExtSems, cudaStream_t stream = 0); 
#line 3970
extern cudaError_t __stdcall cudaDestroyExternalSemaphore(cudaExternalSemaphore_t extSem); 
#line 4037
extern cudaError_t __stdcall cudaLaunchKernel(const void * func, dim3 gridDim, dim3 blockDim, void ** args, size_t sharedMem, cudaStream_t stream); 
#line 4099
extern cudaError_t __stdcall cudaLaunchKernelExC(const cudaLaunchConfig_t * config, const void * func, void ** args); 
#line 4156
extern cudaError_t __stdcall cudaLaunchCooperativeKernel(const void * func, dim3 gridDim, dim3 blockDim, void ** args, size_t sharedMem, cudaStream_t stream); 
#line 4257
__declspec(deprecated) extern cudaError_t __stdcall cudaLaunchCooperativeKernelMultiDevice(cudaLaunchParams * launchParamsList, unsigned numDevices, unsigned flags = 0); 
#line 4304
extern cudaError_t __stdcall cudaFuncSetCacheConfig(const void * func, cudaFuncCache cacheConfig); 
#line 4359
extern cudaError_t __stdcall cudaFuncSetSharedMemConfig(const void * func, cudaSharedMemConfig config); 
#line 4392
extern cudaError_t __stdcall cudaFuncGetAttributes(cudaFuncAttributes * attr, const void * func); 
#line 4429
extern cudaError_t __stdcall cudaFuncSetAttribute(const void * func, cudaFuncAttribute attr, int value); 
#line 4453
__declspec(deprecated) extern cudaError_t __stdcall cudaSetDoubleForDevice(double * d); 
#line 4477
__declspec(deprecated) extern cudaError_t __stdcall cudaSetDoubleForHost(double * d); 
#line 4543
extern cudaError_t __stdcall cudaLaunchHostFunc(cudaStream_t stream, cudaHostFn_t fn, void * userData); 
#line 4600
extern cudaError_t __stdcall cudaOccupancyMaxActiveBlocksPerMultiprocessor(int * numBlocks, const void * func, int blockSize, size_t dynamicSMemSize); 
#line 4629
extern cudaError_t __stdcall cudaOccupancyAvailableDynamicSMemPerBlock(size_t * dynamicSmemSize, const void * func, int numBlocks, int blockSize); 
#line 4674
extern cudaError_t __stdcall cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int * numBlocks, const void * func, int blockSize, size_t dynamicSMemSize, unsigned flags); 
#line 4709
extern cudaError_t __stdcall cudaOccupancyMaxPotentialClusterSize(int * clusterSize, const void * func, const cudaLaunchConfig_t * launchConfig); 
#line 4748
extern cudaError_t __stdcall cudaOccupancyMaxActiveClusters(int * numClusters, const void * func, const cudaLaunchConfig_t * launchConfig); 
#line 4868
extern cudaError_t __stdcall cudaMallocManaged(void ** devPtr, size_t size, unsigned flags = 1); 
#line 4901 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaMalloc(void ** devPtr, size_t size); 
#line 4934
extern cudaError_t __stdcall cudaMallocHost(void ** ptr, size_t size); 
#line 4977
extern cudaError_t __stdcall cudaMallocPitch(void ** devPtr, size_t * pitch, size_t width, size_t height); 
#line 5029
extern cudaError_t __stdcall cudaMallocArray(cudaArray_t * array, const cudaChannelFormatDesc * desc, size_t width, size_t height = 0, unsigned flags = 0); 
#line 5067
extern cudaError_t __stdcall cudaFree(void * devPtr); 
#line 5090
extern cudaError_t __stdcall cudaFreeHost(void * ptr); 
#line 5113
extern cudaError_t __stdcall cudaFreeArray(cudaArray_t array); 
#line 5136
extern cudaError_t __stdcall cudaFreeMipmappedArray(cudaMipmappedArray_t mipmappedArray); 
#line 5202
extern cudaError_t __stdcall cudaHostAlloc(void ** pHost, size_t size, unsigned flags); 
#line 5295
extern cudaError_t __stdcall cudaHostRegister(void * ptr, size_t size, unsigned flags); 
#line 5318
extern cudaError_t __stdcall cudaHostUnregister(void * ptr); 
#line 5363
extern cudaError_t __stdcall cudaHostGetDevicePointer(void ** pDevice, void * pHost, unsigned flags); 
#line 5385
extern cudaError_t __stdcall cudaHostGetFlags(unsigned * pFlags, void * pHost); 
#line 5424
extern cudaError_t __stdcall cudaMalloc3D(cudaPitchedPtr * pitchedDevPtr, cudaExtent extent); 
#line 5569
extern cudaError_t __stdcall cudaMalloc3DArray(cudaArray_t * array, const cudaChannelFormatDesc * desc, cudaExtent extent, unsigned flags = 0); 
#line 5714
extern cudaError_t __stdcall cudaMallocMipmappedArray(cudaMipmappedArray_t * mipmappedArray, const cudaChannelFormatDesc * desc, cudaExtent extent, unsigned numLevels, unsigned flags = 0); 
#line 5747
extern cudaError_t __stdcall cudaGetMipmappedArrayLevel(cudaArray_t * levelArray, cudaMipmappedArray_const_t mipmappedArray, unsigned level); 
#line 5852
extern cudaError_t __stdcall cudaMemcpy3D(const cudaMemcpy3DParms * p); 
#line 5883
extern cudaError_t __stdcall cudaMemcpy3DPeer(const cudaMemcpy3DPeerParms * p); 
#line 6001
extern cudaError_t __stdcall cudaMemcpy3DAsync(const cudaMemcpy3DParms * p, cudaStream_t stream = 0); 
#line 6027
extern cudaError_t __stdcall cudaMemcpy3DPeerAsync(const cudaMemcpy3DPeerParms * p, cudaStream_t stream = 0); 
#line 6061
extern cudaError_t __stdcall cudaMemGetInfo(size_t * free, size_t * total); 
#line 6087
extern cudaError_t __stdcall cudaArrayGetInfo(cudaChannelFormatDesc * desc, cudaExtent * extent, unsigned * flags, cudaArray_t array); 
#line 6116
extern cudaError_t __stdcall cudaArrayGetPlane(cudaArray_t * pPlaneArray, cudaArray_t hArray, unsigned planeIdx); 
#line 6139
extern cudaError_t __stdcall cudaArrayGetMemoryRequirements(cudaArrayMemoryRequirements * memoryRequirements, cudaArray_t array, int device); 
#line 6163
extern cudaError_t __stdcall cudaMipmappedArrayGetMemoryRequirements(cudaArrayMemoryRequirements * memoryRequirements, cudaMipmappedArray_t mipmap, int device); 
#line 6191
extern cudaError_t __stdcall cudaArrayGetSparseProperties(cudaArraySparseProperties * sparseProperties, cudaArray_t array); 
#line 6221 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaMipmappedArrayGetSparseProperties(cudaArraySparseProperties * sparseProperties, cudaMipmappedArray_t mipmap); 
#line 6266 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaMemcpy(void * dst, const void * src, size_t count, cudaMemcpyKind kind); 
#line 6301
extern cudaError_t __stdcall cudaMemcpyPeer(void * dst, int dstDevice, const void * src, int srcDevice, size_t count); 
#line 6350
extern cudaError_t __stdcall cudaMemcpy2D(void * dst, size_t dpitch, const void * src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind); 
#line 6400
extern cudaError_t __stdcall cudaMemcpy2DToArray(cudaArray_t dst, size_t wOffset, size_t hOffset, const void * src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind); 
#line 6450
extern cudaError_t __stdcall cudaMemcpy2DFromArray(void * dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, cudaMemcpyKind kind); 
#line 6497
extern cudaError_t __stdcall cudaMemcpy2DArrayToArray(cudaArray_t dst, size_t wOffsetDst, size_t hOffsetDst, cudaArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t width, size_t height, cudaMemcpyKind kind = cudaMemcpyDeviceToDevice); 
#line 6540
extern cudaError_t __stdcall cudaMemcpyToSymbol(const void * symbol, const void * src, size_t count, size_t offset = 0, cudaMemcpyKind kind = cudaMemcpyHostToDevice); 
#line 6583
extern cudaError_t __stdcall cudaMemcpyFromSymbol(void * dst, const void * symbol, size_t count, size_t offset = 0, cudaMemcpyKind kind = cudaMemcpyDeviceToHost); 
#line 6640
extern cudaError_t __stdcall cudaMemcpyAsync(void * dst, const void * src, size_t count, cudaMemcpyKind kind, cudaStream_t stream = 0); 
#line 6675
extern cudaError_t __stdcall cudaMemcpyPeerAsync(void * dst, int dstDevice, const void * src, int srcDevice, size_t count, cudaStream_t stream = 0); 
#line 6738
extern cudaError_t __stdcall cudaMemcpy2DAsync(void * dst, size_t dpitch, const void * src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream = 0); 
#line 6796
extern cudaError_t __stdcall cudaMemcpy2DToArrayAsync(cudaArray_t dst, size_t wOffset, size_t hOffset, const void * src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream = 0); 
#line 6853
extern cudaError_t __stdcall cudaMemcpy2DFromArrayAsync(void * dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream = 0); 
#line 6904
extern cudaError_t __stdcall cudaMemcpyToSymbolAsync(const void * symbol, const void * src, size_t count, size_t offset, cudaMemcpyKind kind, cudaStream_t stream = 0); 
#line 6955
extern cudaError_t __stdcall cudaMemcpyFromSymbolAsync(void * dst, const void * symbol, size_t count, size_t offset, cudaMemcpyKind kind, cudaStream_t stream = 0); 
#line 6984
extern cudaError_t __stdcall cudaMemset(void * devPtr, int value, size_t count); 
#line 7018
extern cudaError_t __stdcall cudaMemset2D(void * devPtr, size_t pitch, int value, size_t width, size_t height); 
#line 7064
extern cudaError_t __stdcall cudaMemset3D(cudaPitchedPtr pitchedDevPtr, int value, cudaExtent extent); 
#line 7100
extern cudaError_t __stdcall cudaMemsetAsync(void * devPtr, int value, size_t count, cudaStream_t stream = 0); 
#line 7141
extern cudaError_t __stdcall cudaMemset2DAsync(void * devPtr, size_t pitch, int value, size_t width, size_t height, cudaStream_t stream = 0); 
#line 7194
extern cudaError_t __stdcall cudaMemset3DAsync(cudaPitchedPtr pitchedDevPtr, int value, cudaExtent extent, cudaStream_t stream = 0); 
#line 7222
extern cudaError_t __stdcall cudaGetSymbolAddress(void ** devPtr, const void * symbol); 
#line 7249
extern cudaError_t __stdcall cudaGetSymbolSize(size_t * size, const void * symbol); 
#line 7319
extern cudaError_t __stdcall cudaMemPrefetchAsync(const void * devPtr, size_t count, int dstDevice, cudaStream_t stream = 0); 
#line 7435
extern cudaError_t __stdcall cudaMemAdvise(const void * devPtr, size_t count, cudaMemoryAdvise advice, int device); 
#line 7494
extern cudaError_t __stdcall cudaMemRangeGetAttribute(void * data, size_t dataSize, cudaMemRangeAttribute attribute, const void * devPtr, size_t count); 
#line 7533
extern cudaError_t __stdcall cudaMemRangeGetAttributes(void ** data, size_t * dataSizes, cudaMemRangeAttribute * attributes, size_t numAttributes, const void * devPtr, size_t count); 
#line 7593
__declspec(deprecated) extern cudaError_t __stdcall cudaMemcpyToArray(cudaArray_t dst, size_t wOffset, size_t hOffset, const void * src, size_t count, cudaMemcpyKind kind); 
#line 7635
__declspec(deprecated) extern cudaError_t __stdcall cudaMemcpyFromArray(void * dst, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t count, cudaMemcpyKind kind); 
#line 7678
__declspec(deprecated) extern cudaError_t __stdcall cudaMemcpyArrayToArray(cudaArray_t dst, size_t wOffsetDst, size_t hOffsetDst, cudaArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t count, cudaMemcpyKind kind = cudaMemcpyDeviceToDevice); 
#line 7729
__declspec(deprecated) extern cudaError_t __stdcall cudaMemcpyToArrayAsync(cudaArray_t dst, size_t wOffset, size_t hOffset, const void * src, size_t count, cudaMemcpyKind kind, cudaStream_t stream = 0); 
#line 7779
__declspec(deprecated) extern cudaError_t __stdcall cudaMemcpyFromArrayAsync(void * dst, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t count, cudaMemcpyKind kind, cudaStream_t stream = 0); 
#line 7848
extern cudaError_t __stdcall cudaMallocAsync(void ** devPtr, size_t size, cudaStream_t hStream); 
#line 7874
extern cudaError_t __stdcall cudaFreeAsync(void * devPtr, cudaStream_t hStream); 
#line 7899
extern cudaError_t __stdcall cudaMemPoolTrimTo(cudaMemPool_t memPool, size_t minBytesToKeep); 
#line 7943
extern cudaError_t __stdcall cudaMemPoolSetAttribute(cudaMemPool_t memPool, cudaMemPoolAttr attr, void * value); 
#line 7991
extern cudaError_t __stdcall cudaMemPoolGetAttribute(cudaMemPool_t memPool, cudaMemPoolAttr attr, void * value); 
#line 8006
extern cudaError_t __stdcall cudaMemPoolSetAccess(cudaMemPool_t memPool, const cudaMemAccessDesc * descList, size_t count); 
#line 8019
extern cudaError_t __stdcall cudaMemPoolGetAccess(cudaMemAccessFlags * flags, cudaMemPool_t memPool, cudaMemLocation * location); 
#line 8039
extern cudaError_t __stdcall cudaMemPoolCreate(cudaMemPool_t * memPool, const cudaMemPoolProps * poolProps); 
#line 8061
extern cudaError_t __stdcall cudaMemPoolDestroy(cudaMemPool_t memPool); 
#line 8097
extern cudaError_t __stdcall cudaMallocFromPoolAsync(void ** ptr, size_t size, cudaMemPool_t memPool, cudaStream_t stream); 
#line 8122
extern cudaError_t __stdcall cudaMemPoolExportToShareableHandle(void * shareableHandle, cudaMemPool_t memPool, cudaMemAllocationHandleType handleType, unsigned flags); 
#line 8149
extern cudaError_t __stdcall cudaMemPoolImportFromShareableHandle(cudaMemPool_t * memPool, void * shareableHandle, cudaMemAllocationHandleType handleType, unsigned flags); 
#line 8172
extern cudaError_t __stdcall cudaMemPoolExportPointer(cudaMemPoolPtrExportData * exportData, void * ptr); 
#line 8201
extern cudaError_t __stdcall cudaMemPoolImportPointer(void ** ptr, cudaMemPool_t memPool, cudaMemPoolPtrExportData * exportData); 
#line 8353
extern cudaError_t __stdcall cudaPointerGetAttributes(cudaPointerAttributes * attributes, const void * ptr); 
#line 8394
extern cudaError_t __stdcall cudaDeviceCanAccessPeer(int * canAccessPeer, int device, int peerDevice); 
#line 8436
extern cudaError_t __stdcall cudaDeviceEnablePeerAccess(int peerDevice, unsigned flags); 
#line 8458
extern cudaError_t __stdcall cudaDeviceDisablePeerAccess(int peerDevice); 
#line 8522
extern cudaError_t __stdcall cudaGraphicsUnregisterResource(cudaGraphicsResource_t resource); 
#line 8557
extern cudaError_t __stdcall cudaGraphicsResourceSetMapFlags(cudaGraphicsResource_t resource, unsigned flags); 
#line 8596
extern cudaError_t __stdcall cudaGraphicsMapResources(int count, cudaGraphicsResource_t * resources, cudaStream_t stream = 0); 
#line 8631
extern cudaError_t __stdcall cudaGraphicsUnmapResources(int count, cudaGraphicsResource_t * resources, cudaStream_t stream = 0); 
#line 8663
extern cudaError_t __stdcall cudaGraphicsResourceGetMappedPointer(void ** devPtr, size_t * size, cudaGraphicsResource_t resource); 
#line 8701
extern cudaError_t __stdcall cudaGraphicsSubResourceGetMappedArray(cudaArray_t * array, cudaGraphicsResource_t resource, unsigned arrayIndex, unsigned mipLevel); 
#line 8730
extern cudaError_t __stdcall cudaGraphicsResourceGetMappedMipmappedArray(cudaMipmappedArray_t * mipmappedArray, cudaGraphicsResource_t resource); 
#line 8801
__declspec(deprecated) extern cudaError_t __stdcall cudaBindTexture(size_t * offset, const textureReference * texref, const void * devPtr, const cudaChannelFormatDesc * desc, size_t size = 4294967295U); 
#line 8860
__declspec(deprecated) extern cudaError_t __stdcall cudaBindTexture2D(size_t * offset, const textureReference * texref, const void * devPtr, const cudaChannelFormatDesc * desc, size_t width, size_t height, size_t pitch); 
#line 8898
__declspec(deprecated) extern cudaError_t __stdcall cudaBindTextureToArray(const textureReference * texref, cudaArray_const_t array, const cudaChannelFormatDesc * desc); 
#line 8938
__declspec(deprecated) extern cudaError_t __stdcall cudaBindTextureToMipmappedArray(const textureReference * texref, cudaMipmappedArray_const_t mipmappedArray, const cudaChannelFormatDesc * desc); 
#line 8964
__declspec(deprecated) extern cudaError_t __stdcall cudaUnbindTexture(const textureReference * texref); 
#line 8993
__declspec(deprecated) extern cudaError_t __stdcall cudaGetTextureAlignmentOffset(size_t * offset, const textureReference * texref); 
#line 9023
__declspec(deprecated) extern cudaError_t __stdcall cudaGetTextureReference(const textureReference ** texref, const void * symbol); 
#line 9068
__declspec(deprecated) extern cudaError_t __stdcall cudaBindSurfaceToArray(const surfaceReference * surfref, cudaArray_const_t array, const cudaChannelFormatDesc * desc); 
#line 9093
__declspec(deprecated) extern cudaError_t __stdcall cudaGetSurfaceReference(const surfaceReference ** surfref, const void * symbol); 
#line 9128
extern cudaError_t __stdcall cudaGetChannelDesc(cudaChannelFormatDesc * desc, cudaArray_const_t array); 
#line 9158
extern cudaChannelFormatDesc __stdcall cudaCreateChannelDesc(int x, int y, int z, int w, cudaChannelFormatKind f); 
#line 9375
extern cudaError_t __stdcall cudaCreateTextureObject(cudaTextureObject_t * pTexObject, const cudaResourceDesc * pResDesc, const cudaTextureDesc * pTexDesc, const cudaResourceViewDesc * pResViewDesc); 
#line 9599
extern cudaError_t __stdcall cudaCreateTextureObject_v2(cudaTextureObject_t * pTexObject, const cudaResourceDesc * pResDesc, const cudaTextureDesc_v2 * pTexDesc, const cudaResourceViewDesc * pResViewDesc); 
#line 9619
extern cudaError_t __stdcall cudaDestroyTextureObject(cudaTextureObject_t texObject); 
#line 9639
extern cudaError_t __stdcall cudaGetTextureObjectResourceDesc(cudaResourceDesc * pResDesc, cudaTextureObject_t texObject); 
#line 9659
extern cudaError_t __stdcall cudaGetTextureObjectTextureDesc(cudaTextureDesc * pTexDesc, cudaTextureObject_t texObject); 
#line 9679
extern cudaError_t __stdcall cudaGetTextureObjectTextureDesc_v2(cudaTextureDesc_v2 * pTexDesc, cudaTextureObject_t texObject); 
#line 9700
extern cudaError_t __stdcall cudaGetTextureObjectResourceViewDesc(cudaResourceViewDesc * pResViewDesc, cudaTextureObject_t texObject); 
#line 9745
extern cudaError_t __stdcall cudaCreateSurfaceObject(cudaSurfaceObject_t * pSurfObject, const cudaResourceDesc * pResDesc); 
#line 9765
extern cudaError_t __stdcall cudaDestroySurfaceObject(cudaSurfaceObject_t surfObject); 
#line 9784
extern cudaError_t __stdcall cudaGetSurfaceObjectResourceDesc(cudaResourceDesc * pResDesc, cudaSurfaceObject_t surfObject); 
#line 9818
extern cudaError_t __stdcall cudaDriverGetVersion(int * driverVersion); 
#line 9843
extern cudaError_t __stdcall cudaRuntimeGetVersion(int * runtimeVersion); 
#line 9890
extern cudaError_t __stdcall cudaGraphCreate(cudaGraph_t * pGraph, unsigned flags); 
#line 9987
extern cudaError_t __stdcall cudaGraphAddKernelNode(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, const cudaKernelNodeParams * pNodeParams); 
#line 10020
extern cudaError_t __stdcall cudaGraphKernelNodeGetParams(cudaGraphNode_t node, cudaKernelNodeParams * pNodeParams); 
#line 10045
extern cudaError_t __stdcall cudaGraphKernelNodeSetParams(cudaGraphNode_t node, const cudaKernelNodeParams * pNodeParams); 
#line 10065
extern cudaError_t __stdcall cudaGraphKernelNodeCopyAttributes(cudaGraphNode_t hSrc, cudaGraphNode_t hDst); 
#line 10088
extern cudaError_t __stdcall cudaGraphKernelNodeGetAttribute(cudaGraphNode_t hNode, cudaLaunchAttributeID attr, cudaLaunchAttributeValue * value_out); 
#line 10112
extern cudaError_t __stdcall cudaGraphKernelNodeSetAttribute(cudaGraphNode_t hNode, cudaLaunchAttributeID attr, const cudaLaunchAttributeValue * value); 
#line 10162
extern cudaError_t __stdcall cudaGraphAddMemcpyNode(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, const cudaMemcpy3DParms * pCopyParams); 
#line 10221
extern cudaError_t __stdcall cudaGraphAddMemcpyNodeToSymbol(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, const void * symbol, const void * src, size_t count, size_t offset, cudaMemcpyKind kind); 
#line 10290 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaGraphAddMemcpyNodeFromSymbol(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, void * dst, const void * symbol, size_t count, size_t offset, cudaMemcpyKind kind); 
#line 10358 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaGraphAddMemcpyNode1D(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, void * dst, const void * src, size_t count, cudaMemcpyKind kind); 
#line 10390 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaGraphMemcpyNodeGetParams(cudaGraphNode_t node, cudaMemcpy3DParms * pNodeParams); 
#line 10416
extern cudaError_t __stdcall cudaGraphMemcpyNodeSetParams(cudaGraphNode_t node, const cudaMemcpy3DParms * pNodeParams); 
#line 10455
extern cudaError_t __stdcall cudaGraphMemcpyNodeSetParamsToSymbol(cudaGraphNode_t node, const void * symbol, const void * src, size_t count, size_t offset, cudaMemcpyKind kind); 
#line 10501 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaGraphMemcpyNodeSetParamsFromSymbol(cudaGraphNode_t node, void * dst, const void * symbol, size_t count, size_t offset, cudaMemcpyKind kind); 
#line 10547 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaGraphMemcpyNodeSetParams1D(cudaGraphNode_t node, void * dst, const void * src, size_t count, cudaMemcpyKind kind); 
#line 10594 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaGraphAddMemsetNode(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, const cudaMemsetParams * pMemsetParams); 
#line 10617
extern cudaError_t __stdcall cudaGraphMemsetNodeGetParams(cudaGraphNode_t node, cudaMemsetParams * pNodeParams); 
#line 10640
extern cudaError_t __stdcall cudaGraphMemsetNodeSetParams(cudaGraphNode_t node, const cudaMemsetParams * pNodeParams); 
#line 10681
extern cudaError_t __stdcall cudaGraphAddHostNode(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, const cudaHostNodeParams * pNodeParams); 
#line 10704
extern cudaError_t __stdcall cudaGraphHostNodeGetParams(cudaGraphNode_t node, cudaHostNodeParams * pNodeParams); 
#line 10727
extern cudaError_t __stdcall cudaGraphHostNodeSetParams(cudaGraphNode_t node, const cudaHostNodeParams * pNodeParams); 
#line 10767
extern cudaError_t __stdcall cudaGraphAddChildGraphNode(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, cudaGraph_t childGraph); 
#line 10794
extern cudaError_t __stdcall cudaGraphChildGraphNodeGetGraph(cudaGraphNode_t node, cudaGraph_t * pGraph); 
#line 10831
extern cudaError_t __stdcall cudaGraphAddEmptyNode(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies); 
#line 10874
extern cudaError_t __stdcall cudaGraphAddEventRecordNode(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, cudaEvent_t event); 
#line 10901 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaGraphEventRecordNodeGetEvent(cudaGraphNode_t node, cudaEvent_t * event_out); 
#line 10928 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaGraphEventRecordNodeSetEvent(cudaGraphNode_t node, cudaEvent_t event); 
#line 10974 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaGraphAddEventWaitNode(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, cudaEvent_t event); 
#line 11001 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaGraphEventWaitNodeGetEvent(cudaGraphNode_t node, cudaEvent_t * event_out); 
#line 11028 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaGraphEventWaitNodeSetEvent(cudaGraphNode_t node, cudaEvent_t event); 
#line 11077 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaGraphAddExternalSemaphoresSignalNode(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, const cudaExternalSemaphoreSignalNodeParams * nodeParams); 
#line 11110 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaGraphExternalSemaphoresSignalNodeGetParams(cudaGraphNode_t hNode, cudaExternalSemaphoreSignalNodeParams * params_out); 
#line 11137 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaGraphExternalSemaphoresSignalNodeSetParams(cudaGraphNode_t hNode, const cudaExternalSemaphoreSignalNodeParams * nodeParams); 
#line 11186 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaGraphAddExternalSemaphoresWaitNode(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, const cudaExternalSemaphoreWaitNodeParams * nodeParams); 
#line 11219 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaGraphExternalSemaphoresWaitNodeGetParams(cudaGraphNode_t hNode, cudaExternalSemaphoreWaitNodeParams * params_out); 
#line 11246 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaGraphExternalSemaphoresWaitNodeSetParams(cudaGraphNode_t hNode, const cudaExternalSemaphoreWaitNodeParams * nodeParams); 
#line 11323 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaGraphAddMemAllocNode(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, cudaMemAllocNodeParams * nodeParams); 
#line 11350 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaGraphMemAllocNodeGetParams(cudaGraphNode_t node, cudaMemAllocNodeParams * params_out); 
#line 11410 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaGraphAddMemFreeNode(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, void * dptr); 
#line 11434 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaGraphMemFreeNodeGetParams(cudaGraphNode_t node, void * dptr_out); 
#line 11462 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaDeviceGraphMemTrim(int device); 
#line 11499 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaDeviceGetGraphMemAttribute(int device, cudaGraphMemAttributeType attr, void * value); 
#line 11533 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaDeviceSetGraphMemAttribute(int device, cudaGraphMemAttributeType attr, void * value); 
#line 11561 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaGraphClone(cudaGraph_t * pGraphClone, cudaGraph_t originalGraph); 
#line 11589
extern cudaError_t __stdcall cudaGraphNodeFindInClone(cudaGraphNode_t * pNode, cudaGraphNode_t originalNode, cudaGraph_t clonedGraph); 
#line 11620
extern cudaError_t __stdcall cudaGraphNodeGetType(cudaGraphNode_t node, cudaGraphNodeType * pType); 
#line 11651
extern cudaError_t __stdcall cudaGraphGetNodes(cudaGraph_t graph, cudaGraphNode_t * nodes, size_t * numNodes); 
#line 11682
extern cudaError_t __stdcall cudaGraphGetRootNodes(cudaGraph_t graph, cudaGraphNode_t * pRootNodes, size_t * pNumRootNodes); 
#line 11716
extern cudaError_t __stdcall cudaGraphGetEdges(cudaGraph_t graph, cudaGraphNode_t * from, cudaGraphNode_t * to, size_t * numEdges); 
#line 11747
extern cudaError_t __stdcall cudaGraphNodeGetDependencies(cudaGraphNode_t node, cudaGraphNode_t * pDependencies, size_t * pNumDependencies); 
#line 11779
extern cudaError_t __stdcall cudaGraphNodeGetDependentNodes(cudaGraphNode_t node, cudaGraphNode_t * pDependentNodes, size_t * pNumDependentNodes); 
#line 11810
extern cudaError_t __stdcall cudaGraphAddDependencies(cudaGraph_t graph, const cudaGraphNode_t * from, const cudaGraphNode_t * to, size_t numDependencies); 
#line 11841
extern cudaError_t __stdcall cudaGraphRemoveDependencies(cudaGraph_t graph, const cudaGraphNode_t * from, const cudaGraphNode_t * to, size_t numDependencies); 
#line 11871
extern cudaError_t __stdcall cudaGraphDestroyNode(cudaGraphNode_t node); 
#line 11909
extern cudaError_t __stdcall cudaGraphInstantiate(cudaGraphExec_t * pGraphExec, cudaGraph_t graph, cudaGraphNode_t * pErrorNode, char * pLogBuffer, size_t bufferSize); 
#line 11957
extern cudaError_t __stdcall cudaGraphInstantiateWithFlags(cudaGraphExec_t * pGraphExec, cudaGraph_t graph, unsigned __int64 flags); 
#line 12001 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaGraphExecKernelNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const cudaKernelNodeParams * pNodeParams); 
#line 12051
extern cudaError_t __stdcall cudaGraphExecMemcpyNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const cudaMemcpy3DParms * pNodeParams); 
#line 12106
extern cudaError_t __stdcall cudaGraphExecMemcpyNodeSetParamsToSymbol(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const void * symbol, const void * src, size_t count, size_t offset, cudaMemcpyKind kind); 
#line 12169 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaGraphExecMemcpyNodeSetParamsFromSymbol(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, void * dst, const void * symbol, size_t count, size_t offset, cudaMemcpyKind kind); 
#line 12230 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaGraphExecMemcpyNodeSetParams1D(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, void * dst, const void * src, size_t count, cudaMemcpyKind kind); 
#line 12284 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaGraphExecMemsetNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const cudaMemsetParams * pNodeParams); 
#line 12323
extern cudaError_t __stdcall cudaGraphExecHostNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const cudaHostNodeParams * pNodeParams); 
#line 12369
extern cudaError_t __stdcall cudaGraphExecChildGraphNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, cudaGraph_t childGraph); 
#line 12413 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaGraphExecEventRecordNodeSetEvent(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, cudaEvent_t event); 
#line 12457 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaGraphExecEventWaitNodeSetEvent(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, cudaEvent_t event); 
#line 12504 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaGraphExecExternalSemaphoresSignalNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, const cudaExternalSemaphoreSignalNodeParams * nodeParams); 
#line 12551 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaGraphExecExternalSemaphoresWaitNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, const cudaExternalSemaphoreWaitNodeParams * nodeParams); 
#line 12591 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaGraphNodeSetEnabled(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, unsigned isEnabled); 
#line 12625 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaGraphNodeGetEnabled(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, unsigned * isEnabled); 
#line 12706 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaGraphExecUpdate(cudaGraphExec_t hGraphExec, cudaGraph_t hGraph, cudaGraphNode_t * hErrorNode_out, cudaGraphExecUpdateResult * updateResult_out); 
#line 12731
extern cudaError_t __stdcall cudaGraphUpload(cudaGraphExec_t graphExec, cudaStream_t stream); 
#line 12762 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaGraphLaunch(cudaGraphExec_t graphExec, cudaStream_t stream); 
#line 12785
extern cudaError_t __stdcall cudaGraphExecDestroy(cudaGraphExec_t graphExec); 
#line 12806
extern cudaError_t __stdcall cudaGraphDestroy(cudaGraph_t graph); 
#line 12825
extern cudaError_t __stdcall cudaGraphDebugDotPrint(cudaGraph_t graph, const char * path, unsigned flags); 
#line 12861
extern cudaError_t __stdcall cudaUserObjectCreate(cudaUserObject_t * object_out, void * ptr, cudaHostFn_t destroy, unsigned initialRefcount, unsigned flags); 
#line 12885
extern cudaError_t __stdcall cudaUserObjectRetain(cudaUserObject_t object, unsigned count = 1); 
#line 12913
extern cudaError_t __stdcall cudaUserObjectRelease(cudaUserObject_t object, unsigned count = 1); 
#line 12941
extern cudaError_t __stdcall cudaGraphRetainUserObject(cudaGraph_t graph, cudaUserObject_t object, unsigned count = 1, unsigned flags = 0); 
#line 12966
extern cudaError_t __stdcall cudaGraphReleaseUserObject(cudaGraph_t graph, cudaUserObject_t object, unsigned count = 1); 
#line 13032
extern cudaError_t __stdcall cudaGetDriverEntryPoint(const char * symbol, void ** funcPtr, unsigned __int64 flags); 
#line 13037
extern cudaError_t __stdcall cudaGetExportTable(const void ** ppExportTable, const cudaUUID_t * pExportTableId); 
#line 13213
extern cudaError_t __cdecl cudaGetFuncBySymbol(cudaFunction_t * functionPtr, const void * symbolPtr); 
#line 13365 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\cuda_runtime_api.h"
}
#line 124 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\channel_descriptor.h"
template< class T> __inline ::cudaChannelFormatDesc cudaCreateChannelDesc() 
#line 125
{ 
#line 126
return cudaCreateChannelDesc(0, 0, 0, 0, cudaChannelFormatKindNone); 
#line 127
} 
#line 129
static __inline cudaChannelFormatDesc cudaCreateChannelDescHalf() 
#line 130
{ 
#line 131
int e = (((int)sizeof(unsigned short)) * 8); 
#line 133
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindFloat); 
#line 134
} 
#line 136
static __inline cudaChannelFormatDesc cudaCreateChannelDescHalf1() 
#line 137
{ 
#line 138
int e = (((int)sizeof(unsigned short)) * 8); 
#line 140
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindFloat); 
#line 141
} 
#line 143
static __inline cudaChannelFormatDesc cudaCreateChannelDescHalf2() 
#line 144
{ 
#line 145
int e = (((int)sizeof(unsigned short)) * 8); 
#line 147
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindFloat); 
#line 148
} 
#line 150
static __inline cudaChannelFormatDesc cudaCreateChannelDescHalf4() 
#line 151
{ 
#line 152
int e = (((int)sizeof(unsigned short)) * 8); 
#line 154
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindFloat); 
#line 155
} 
#line 157
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< char> () 
#line 158
{ 
#line 159
int e = (((int)sizeof(char)) * 8); 
#line 164 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\channel_descriptor.h"
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned); 
#line 166 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\channel_descriptor.h"
} 
#line 168
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< signed char> () 
#line 169
{ 
#line 170
int e = (((int)sizeof(signed char)) * 8); 
#line 172
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned); 
#line 173
} 
#line 175
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< unsigned char> () 
#line 176
{ 
#line 177
int e = (((int)sizeof(unsigned char)) * 8); 
#line 179
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned); 
#line 180
} 
#line 182
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< char1> () 
#line 183
{ 
#line 184
int e = (((int)sizeof(signed char)) * 8); 
#line 186
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned); 
#line 187
} 
#line 189
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< uchar1> () 
#line 190
{ 
#line 191
int e = (((int)sizeof(unsigned char)) * 8); 
#line 193
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned); 
#line 194
} 
#line 196
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< char2> () 
#line 197
{ 
#line 198
int e = (((int)sizeof(signed char)) * 8); 
#line 200
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindSigned); 
#line 201
} 
#line 203
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< uchar2> () 
#line 204
{ 
#line 205
int e = (((int)sizeof(unsigned char)) * 8); 
#line 207
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindUnsigned); 
#line 208
} 
#line 210
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< char4> () 
#line 211
{ 
#line 212
int e = (((int)sizeof(signed char)) * 8); 
#line 214
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindSigned); 
#line 215
} 
#line 217
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< uchar4> () 
#line 218
{ 
#line 219
int e = (((int)sizeof(unsigned char)) * 8); 
#line 221
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindUnsigned); 
#line 222
} 
#line 224
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< short> () 
#line 225
{ 
#line 226
int e = (((int)sizeof(short)) * 8); 
#line 228
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned); 
#line 229
} 
#line 231
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< unsigned short> () 
#line 232
{ 
#line 233
int e = (((int)sizeof(unsigned short)) * 8); 
#line 235
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned); 
#line 236
} 
#line 238
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< short1> () 
#line 239
{ 
#line 240
int e = (((int)sizeof(short)) * 8); 
#line 242
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned); 
#line 243
} 
#line 245
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< ushort1> () 
#line 246
{ 
#line 247
int e = (((int)sizeof(unsigned short)) * 8); 
#line 249
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned); 
#line 250
} 
#line 252
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< short2> () 
#line 253
{ 
#line 254
int e = (((int)sizeof(short)) * 8); 
#line 256
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindSigned); 
#line 257
} 
#line 259
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< ushort2> () 
#line 260
{ 
#line 261
int e = (((int)sizeof(unsigned short)) * 8); 
#line 263
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindUnsigned); 
#line 264
} 
#line 266
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< short4> () 
#line 267
{ 
#line 268
int e = (((int)sizeof(short)) * 8); 
#line 270
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindSigned); 
#line 271
} 
#line 273
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< ushort4> () 
#line 274
{ 
#line 275
int e = (((int)sizeof(unsigned short)) * 8); 
#line 277
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindUnsigned); 
#line 278
} 
#line 280
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< int> () 
#line 281
{ 
#line 282
int e = (((int)sizeof(int)) * 8); 
#line 284
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned); 
#line 285
} 
#line 287
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< unsigned> () 
#line 288
{ 
#line 289
int e = (((int)sizeof(unsigned)) * 8); 
#line 291
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned); 
#line 292
} 
#line 294
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< int1> () 
#line 295
{ 
#line 296
int e = (((int)sizeof(int)) * 8); 
#line 298
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned); 
#line 299
} 
#line 301
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< uint1> () 
#line 302
{ 
#line 303
int e = (((int)sizeof(unsigned)) * 8); 
#line 305
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned); 
#line 306
} 
#line 308
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< int2> () 
#line 309
{ 
#line 310
int e = (((int)sizeof(int)) * 8); 
#line 312
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindSigned); 
#line 313
} 
#line 315
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< uint2> () 
#line 316
{ 
#line 317
int e = (((int)sizeof(unsigned)) * 8); 
#line 319
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindUnsigned); 
#line 320
} 
#line 322
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< int4> () 
#line 323
{ 
#line 324
int e = (((int)sizeof(int)) * 8); 
#line 326
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindSigned); 
#line 327
} 
#line 329
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< uint4> () 
#line 330
{ 
#line 331
int e = (((int)sizeof(unsigned)) * 8); 
#line 333
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindUnsigned); 
#line 334
} 
#line 338
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< long> () 
#line 339
{ 
#line 340
int e = (((int)sizeof(long)) * 8); 
#line 342
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned); 
#line 343
} 
#line 345
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< unsigned long> () 
#line 346
{ 
#line 347
int e = (((int)sizeof(unsigned long)) * 8); 
#line 349
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned); 
#line 350
} 
#line 352
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< long1> () 
#line 353
{ 
#line 354
int e = (((int)sizeof(long)) * 8); 
#line 356
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned); 
#line 357
} 
#line 359
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< ulong1> () 
#line 360
{ 
#line 361
int e = (((int)sizeof(unsigned long)) * 8); 
#line 363
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned); 
#line 364
} 
#line 366
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< long2> () 
#line 367
{ 
#line 368
int e = (((int)sizeof(long)) * 8); 
#line 370
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindSigned); 
#line 371
} 
#line 373
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< ulong2> () 
#line 374
{ 
#line 375
int e = (((int)sizeof(unsigned long)) * 8); 
#line 377
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindUnsigned); 
#line 378
} 
#line 380
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< long4> () 
#line 381
{ 
#line 382
int e = (((int)sizeof(long)) * 8); 
#line 384
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindSigned); 
#line 385
} 
#line 387
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< ulong4> () 
#line 388
{ 
#line 389
int e = (((int)sizeof(unsigned long)) * 8); 
#line 391
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindUnsigned); 
#line 392
} 
#line 396 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\channel_descriptor.h"
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< float> () 
#line 397
{ 
#line 398
int e = (((int)sizeof(float)) * 8); 
#line 400
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindFloat); 
#line 401
} 
#line 403
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< float1> () 
#line 404
{ 
#line 405
int e = (((int)sizeof(float)) * 8); 
#line 407
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindFloat); 
#line 408
} 
#line 410
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< float2> () 
#line 411
{ 
#line 412
int e = (((int)sizeof(float)) * 8); 
#line 414
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindFloat); 
#line 415
} 
#line 417
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< float4> () 
#line 418
{ 
#line 419
int e = (((int)sizeof(float)) * 8); 
#line 421
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindFloat); 
#line 422
} 
#line 424
static __inline cudaChannelFormatDesc cudaCreateChannelDescNV12() 
#line 425
{ 
#line 426
int e = (((int)sizeof(char)) * 8); 
#line 428
return cudaCreateChannelDesc(e, e, e, 0, cudaChannelFormatKindNV12); 
#line 429
} 
#line 431
template< cudaChannelFormatKind > __inline ::cudaChannelFormatDesc cudaCreateChannelDesc() 
#line 432
{ 
#line 433
return cudaCreateChannelDesc(0, 0, 0, 0, cudaChannelFormatKindNone); 
#line 434
} 
#line 437
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindSignedNormalized8X1> () 
#line 438
{ 
#line 439
return cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindSignedNormalized8X1); 
#line 440
} 
#line 442
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindSignedNormalized8X2> () 
#line 443
{ 
#line 444
return cudaCreateChannelDesc(8, 8, 0, 0, cudaChannelFormatKindSignedNormalized8X2); 
#line 445
} 
#line 447
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindSignedNormalized8X4> () 
#line 448
{ 
#line 449
return cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindSignedNormalized8X4); 
#line 450
} 
#line 453
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindUnsignedNormalized8X1> () 
#line 454
{ 
#line 455
return cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsignedNormalized8X1); 
#line 456
} 
#line 458
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindUnsignedNormalized8X2> () 
#line 459
{ 
#line 460
return cudaCreateChannelDesc(8, 8, 0, 0, cudaChannelFormatKindUnsignedNormalized8X2); 
#line 461
} 
#line 463
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindUnsignedNormalized8X4> () 
#line 464
{ 
#line 465
return cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsignedNormalized8X4); 
#line 466
} 
#line 469
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindSignedNormalized16X1> () 
#line 470
{ 
#line 471
return cudaCreateChannelDesc(16, 0, 0, 0, cudaChannelFormatKindSignedNormalized16X1); 
#line 472
} 
#line 474
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindSignedNormalized16X2> () 
#line 475
{ 
#line 476
return cudaCreateChannelDesc(16, 16, 0, 0, cudaChannelFormatKindSignedNormalized16X2); 
#line 477
} 
#line 479
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindSignedNormalized16X4> () 
#line 480
{ 
#line 481
return cudaCreateChannelDesc(16, 16, 16, 16, cudaChannelFormatKindSignedNormalized16X4); 
#line 482
} 
#line 485
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindUnsignedNormalized16X1> () 
#line 486
{ 
#line 487
return cudaCreateChannelDesc(16, 0, 0, 0, cudaChannelFormatKindUnsignedNormalized16X1); 
#line 488
} 
#line 490
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindUnsignedNormalized16X2> () 
#line 491
{ 
#line 492
return cudaCreateChannelDesc(16, 16, 0, 0, cudaChannelFormatKindUnsignedNormalized16X2); 
#line 493
} 
#line 495
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindUnsignedNormalized16X4> () 
#line 496
{ 
#line 497
return cudaCreateChannelDesc(16, 16, 16, 16, cudaChannelFormatKindUnsignedNormalized16X4); 
#line 498
} 
#line 501
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindNV12> () 
#line 502
{ 
#line 503
return cudaCreateChannelDesc(8, 8, 8, 0, cudaChannelFormatKindNV12); 
#line 504
} 
#line 507
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindUnsignedBlockCompressed1> () 
#line 508
{ 
#line 509
return cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsignedBlockCompressed1); 
#line 510
} 
#line 513
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindUnsignedBlockCompressed1SRGB> () 
#line 514
{ 
#line 515
return cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsignedBlockCompressed1SRGB); 
#line 516
} 
#line 519
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindUnsignedBlockCompressed2> () 
#line 520
{ 
#line 521
return cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsignedBlockCompressed2); 
#line 522
} 
#line 525
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindUnsignedBlockCompressed2SRGB> () 
#line 526
{ 
#line 527
return cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsignedBlockCompressed2SRGB); 
#line 528
} 
#line 531
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindUnsignedBlockCompressed3> () 
#line 532
{ 
#line 533
return cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsignedBlockCompressed3); 
#line 534
} 
#line 537
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindUnsignedBlockCompressed3SRGB> () 
#line 538
{ 
#line 539
return cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsignedBlockCompressed3SRGB); 
#line 540
} 
#line 543
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindUnsignedBlockCompressed4> () 
#line 544
{ 
#line 545
return cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsignedBlockCompressed4); 
#line 546
} 
#line 549
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindSignedBlockCompressed4> () 
#line 550
{ 
#line 551
return cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindSignedBlockCompressed4); 
#line 552
} 
#line 555
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindUnsignedBlockCompressed5> () 
#line 556
{ 
#line 557
return cudaCreateChannelDesc(8, 8, 0, 0, cudaChannelFormatKindUnsignedBlockCompressed5); 
#line 558
} 
#line 561
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindSignedBlockCompressed5> () 
#line 562
{ 
#line 563
return cudaCreateChannelDesc(8, 8, 0, 0, cudaChannelFormatKindSignedBlockCompressed5); 
#line 564
} 
#line 567
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindUnsignedBlockCompressed6H> () 
#line 568
{ 
#line 569
return cudaCreateChannelDesc(16, 16, 16, 0, cudaChannelFormatKindUnsignedBlockCompressed6H); 
#line 570
} 
#line 573
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindSignedBlockCompressed6H> () 
#line 574
{ 
#line 575
return cudaCreateChannelDesc(16, 16, 16, 0, cudaChannelFormatKindSignedBlockCompressed6H); 
#line 576
} 
#line 579
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindUnsignedBlockCompressed7> () 
#line 580
{ 
#line 581
return cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsignedBlockCompressed7); 
#line 582
} 
#line 585
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< cudaChannelFormatKindUnsignedBlockCompressed7SRGB> () 
#line 586
{ 
#line 587
return cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsignedBlockCompressed7SRGB); 
#line 588
} 
#line 79 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\driver_functions.h"
static __inline cudaPitchedPtr make_cudaPitchedPtr(void *d, size_t p, size_t xsz, size_t ysz) 
#line 80
{ 
#line 81
cudaPitchedPtr s; 
#line 83
(s.ptr) = d; 
#line 84
(s.pitch) = p; 
#line 85
(s.xsize) = xsz; 
#line 86
(s.ysize) = ysz; 
#line 88
return s; 
#line 89
} 
#line 106
static __inline cudaPos make_cudaPos(size_t x, size_t y, size_t z) 
#line 107
{ 
#line 108
cudaPos p; 
#line 110
(p.x) = x; 
#line 111
(p.y) = y; 
#line 112
(p.z) = z; 
#line 114
return p; 
#line 115
} 
#line 132
static __inline cudaExtent make_cudaExtent(size_t w, size_t h, size_t d) 
#line 133
{ 
#line 134
cudaExtent e; 
#line 136
(e.width) = w; 
#line 137
(e.height) = h; 
#line 138
(e.depth) = d; 
#line 140
return e; 
#line 141
} 
#line 73 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\vector_functions.h"
static __inline char1 make_char1(signed char x); 
#line 75
static __inline uchar1 make_uchar1(unsigned char x); 
#line 77
static __inline char2 make_char2(signed char x, signed char y); 
#line 79
static __inline uchar2 make_uchar2(unsigned char x, unsigned char y); 
#line 81
static __inline char3 make_char3(signed char x, signed char y, signed char z); 
#line 83
static __inline uchar3 make_uchar3(unsigned char x, unsigned char y, unsigned char z); 
#line 85
static __inline char4 make_char4(signed char x, signed char y, signed char z, signed char w); 
#line 87
static __inline uchar4 make_uchar4(unsigned char x, unsigned char y, unsigned char z, unsigned char w); 
#line 89
static __inline short1 make_short1(short x); 
#line 91
static __inline ushort1 make_ushort1(unsigned short x); 
#line 93
static __inline short2 make_short2(short x, short y); 
#line 95
static __inline ushort2 make_ushort2(unsigned short x, unsigned short y); 
#line 97
static __inline short3 make_short3(short x, short y, short z); 
#line 99
static __inline ushort3 make_ushort3(unsigned short x, unsigned short y, unsigned short z); 
#line 101
static __inline short4 make_short4(short x, short y, short z, short w); 
#line 103
static __inline ushort4 make_ushort4(unsigned short x, unsigned short y, unsigned short z, unsigned short w); 
#line 105
static __inline int1 make_int1(int x); 
#line 107
static __inline uint1 make_uint1(unsigned x); 
#line 109
static __inline int2 make_int2(int x, int y); 
#line 111
static __inline uint2 make_uint2(unsigned x, unsigned y); 
#line 113
static __inline int3 make_int3(int x, int y, int z); 
#line 115
static __inline uint3 make_uint3(unsigned x, unsigned y, unsigned z); 
#line 117
static __inline int4 make_int4(int x, int y, int z, int w); 
#line 119
static __inline uint4 make_uint4(unsigned x, unsigned y, unsigned z, unsigned w); 
#line 121
static __inline long1 make_long1(long x); 
#line 123
static __inline ulong1 make_ulong1(unsigned long x); 
#line 125
static __inline long2 make_long2(long x, long y); 
#line 127
static __inline ulong2 make_ulong2(unsigned long x, unsigned long y); 
#line 129
static __inline long3 make_long3(long x, long y, long z); 
#line 131
static __inline ulong3 make_ulong3(unsigned long x, unsigned long y, unsigned long z); 
#line 133
static __inline long4 make_long4(long x, long y, long z, long w); 
#line 135
static __inline ulong4 make_ulong4(unsigned long x, unsigned long y, unsigned long z, unsigned long w); 
#line 137
static __inline float1 make_float1(float x); 
#line 139
static __inline float2 make_float2(float x, float y); 
#line 141
static __inline float3 make_float3(float x, float y, float z); 
#line 143
static __inline float4 make_float4(float x, float y, float z, float w); 
#line 145
static __inline longlong1 make_longlong1(__int64 x); 
#line 147
static __inline ulonglong1 make_ulonglong1(unsigned __int64 x); 
#line 149
static __inline longlong2 make_longlong2(__int64 x, __int64 y); 
#line 151
static __inline ulonglong2 make_ulonglong2(unsigned __int64 x, unsigned __int64 y); 
#line 153
static __inline longlong3 make_longlong3(__int64 x, __int64 y, __int64 z); 
#line 155
static __inline ulonglong3 make_ulonglong3(unsigned __int64 x, unsigned __int64 y, unsigned __int64 z); 
#line 157
static __inline longlong4 make_longlong4(__int64 x, __int64 y, __int64 z, __int64 w); 
#line 159
static __inline ulonglong4 make_ulonglong4(unsigned __int64 x, unsigned __int64 y, unsigned __int64 z, unsigned __int64 w); 
#line 161
static __inline double1 make_double1(double x); 
#line 163
static __inline double2 make_double2(double x, double y); 
#line 165
static __inline double3 make_double3(double x, double y, double z); 
#line 167
static __inline double4 make_double4(double x, double y, double z, double w); 
#line 73 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\vector_functions.hpp"
static __inline char1 make_char1(signed char x) 
#line 74
{ 
#line 75
char1 t; (t.x) = x; return t; 
#line 76
} 
#line 78
static __inline uchar1 make_uchar1(unsigned char x) 
#line 79
{ 
#line 80
uchar1 t; (t.x) = x; return t; 
#line 81
} 
#line 83
static __inline char2 make_char2(signed char x, signed char y) 
#line 84
{ 
#line 85
char2 t; (t.x) = x; (t.y) = y; return t; 
#line 86
} 
#line 88
static __inline uchar2 make_uchar2(unsigned char x, unsigned char y) 
#line 89
{ 
#line 90
uchar2 t; (t.x) = x; (t.y) = y; return t; 
#line 91
} 
#line 93
static __inline char3 make_char3(signed char x, signed char y, signed char z) 
#line 94
{ 
#line 95
char3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
#line 96
} 
#line 98
static __inline uchar3 make_uchar3(unsigned char x, unsigned char y, unsigned char z) 
#line 99
{ 
#line 100
uchar3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
#line 101
} 
#line 103
static __inline char4 make_char4(signed char x, signed char y, signed char z, signed char w) 
#line 104
{ 
#line 105
char4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
#line 106
} 
#line 108
static __inline uchar4 make_uchar4(unsigned char x, unsigned char y, unsigned char z, unsigned char w) 
#line 109
{ 
#line 110
uchar4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
#line 111
} 
#line 113
static __inline short1 make_short1(short x) 
#line 114
{ 
#line 115
short1 t; (t.x) = x; return t; 
#line 116
} 
#line 118
static __inline ushort1 make_ushort1(unsigned short x) 
#line 119
{ 
#line 120
ushort1 t; (t.x) = x; return t; 
#line 121
} 
#line 123
static __inline short2 make_short2(short x, short y) 
#line 124
{ 
#line 125
short2 t; (t.x) = x; (t.y) = y; return t; 
#line 126
} 
#line 128
static __inline ushort2 make_ushort2(unsigned short x, unsigned short y) 
#line 129
{ 
#line 130
ushort2 t; (t.x) = x; (t.y) = y; return t; 
#line 131
} 
#line 133
static __inline short3 make_short3(short x, short y, short z) 
#line 134
{ 
#line 135
short3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
#line 136
} 
#line 138
static __inline ushort3 make_ushort3(unsigned short x, unsigned short y, unsigned short z) 
#line 139
{ 
#line 140
ushort3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
#line 141
} 
#line 143
static __inline short4 make_short4(short x, short y, short z, short w) 
#line 144
{ 
#line 145
short4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
#line 146
} 
#line 148
static __inline ushort4 make_ushort4(unsigned short x, unsigned short y, unsigned short z, unsigned short w) 
#line 149
{ 
#line 150
ushort4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
#line 151
} 
#line 153
static __inline int1 make_int1(int x) 
#line 154
{ 
#line 155
int1 t; (t.x) = x; return t; 
#line 156
} 
#line 158
static __inline uint1 make_uint1(unsigned x) 
#line 159
{ 
#line 160
uint1 t; (t.x) = x; return t; 
#line 161
} 
#line 163
static __inline int2 make_int2(int x, int y) 
#line 164
{ 
#line 165
int2 t; (t.x) = x; (t.y) = y; return t; 
#line 166
} 
#line 168
static __inline uint2 make_uint2(unsigned x, unsigned y) 
#line 169
{ 
#line 170
uint2 t; (t.x) = x; (t.y) = y; return t; 
#line 171
} 
#line 173
static __inline int3 make_int3(int x, int y, int z) 
#line 174
{ 
#line 175
int3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
#line 176
} 
#line 178
static __inline uint3 make_uint3(unsigned x, unsigned y, unsigned z) 
#line 179
{ 
#line 180
uint3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
#line 181
} 
#line 183
static __inline int4 make_int4(int x, int y, int z, int w) 
#line 184
{ 
#line 185
int4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
#line 186
} 
#line 188
static __inline uint4 make_uint4(unsigned x, unsigned y, unsigned z, unsigned w) 
#line 189
{ 
#line 190
uint4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
#line 191
} 
#line 193
static __inline long1 make_long1(long x) 
#line 194
{ 
#line 195
long1 t; (t.x) = x; return t; 
#line 196
} 
#line 198
static __inline ulong1 make_ulong1(unsigned long x) 
#line 199
{ 
#line 200
ulong1 t; (t.x) = x; return t; 
#line 201
} 
#line 203
static __inline long2 make_long2(long x, long y) 
#line 204
{ 
#line 205
long2 t; (t.x) = x; (t.y) = y; return t; 
#line 206
} 
#line 208
static __inline ulong2 make_ulong2(unsigned long x, unsigned long y) 
#line 209
{ 
#line 210
ulong2 t; (t.x) = x; (t.y) = y; return t; 
#line 211
} 
#line 213
static __inline long3 make_long3(long x, long y, long z) 
#line 214
{ 
#line 215
long3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
#line 216
} 
#line 218
static __inline ulong3 make_ulong3(unsigned long x, unsigned long y, unsigned long z) 
#line 219
{ 
#line 220
ulong3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
#line 221
} 
#line 223
static __inline long4 make_long4(long x, long y, long z, long w) 
#line 224
{ 
#line 225
long4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
#line 226
} 
#line 228
static __inline ulong4 make_ulong4(unsigned long x, unsigned long y, unsigned long z, unsigned long w) 
#line 229
{ 
#line 230
ulong4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
#line 231
} 
#line 233
static __inline float1 make_float1(float x) 
#line 234
{ 
#line 235
float1 t; (t.x) = x; return t; 
#line 236
} 
#line 238
static __inline float2 make_float2(float x, float y) 
#line 239
{ 
#line 240
float2 t; (t.x) = x; (t.y) = y; return t; 
#line 241
} 
#line 243
static __inline float3 make_float3(float x, float y, float z) 
#line 244
{ 
#line 245
float3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
#line 246
} 
#line 248
static __inline float4 make_float4(float x, float y, float z, float w) 
#line 249
{ 
#line 250
float4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
#line 251
} 
#line 253
static __inline longlong1 make_longlong1(__int64 x) 
#line 254
{ 
#line 255
longlong1 t; (t.x) = x; return t; 
#line 256
} 
#line 258
static __inline ulonglong1 make_ulonglong1(unsigned __int64 x) 
#line 259
{ 
#line 260
ulonglong1 t; (t.x) = x; return t; 
#line 261
} 
#line 263
static __inline longlong2 make_longlong2(__int64 x, __int64 y) 
#line 264
{ 
#line 265
longlong2 t; (t.x) = x; (t.y) = y; return t; 
#line 266
} 
#line 268
static __inline ulonglong2 make_ulonglong2(unsigned __int64 x, unsigned __int64 y) 
#line 269
{ 
#line 270
ulonglong2 t; (t.x) = x; (t.y) = y; return t; 
#line 271
} 
#line 273
static __inline longlong3 make_longlong3(__int64 x, __int64 y, __int64 z) 
#line 274
{ 
#line 275
longlong3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
#line 276
} 
#line 278
static __inline ulonglong3 make_ulonglong3(unsigned __int64 x, unsigned __int64 y, unsigned __int64 z) 
#line 279
{ 
#line 280
ulonglong3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
#line 281
} 
#line 283
static __inline longlong4 make_longlong4(__int64 x, __int64 y, __int64 z, __int64 w) 
#line 284
{ 
#line 285
longlong4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
#line 286
} 
#line 288
static __inline ulonglong4 make_ulonglong4(unsigned __int64 x, unsigned __int64 y, unsigned __int64 z, unsigned __int64 w) 
#line 289
{ 
#line 290
ulonglong4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
#line 291
} 
#line 293
static __inline double1 make_double1(double x) 
#line 294
{ 
#line 295
double1 t; (t.x) = x; return t; 
#line 296
} 
#line 298
static __inline double2 make_double2(double x, double y) 
#line 299
{ 
#line 300
double2 t; (t.x) = x; (t.y) = y; return t; 
#line 301
} 
#line 303
static __inline double3 make_double3(double x, double y, double z) 
#line 304
{ 
#line 305
double3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
#line 306
} 
#line 308
static __inline double4 make_double4(double x, double y, double z, double w) 
#line 309
{ 
#line 310
double4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
#line 311
} 
#line 14 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\errno.h"
#pragma warning(push)
#pragma warning(disable: 4324 4514 4574 4710 4793 4820 4995 4996 28719 28726 28727 )
#line 18
__pragma( pack ( push, 8 )) extern "C" {
#line 23
int *__cdecl _errno(); 
#line 26
errno_t __cdecl _set_errno(int _Value); 
#line 27
errno_t __cdecl _get_errno(int * _Value); 
#line 29
unsigned long *__cdecl __doserrno(); 
#line 32
errno_t __cdecl _set_doserrno(unsigned long _Value); 
#line 33
errno_t __cdecl _get_doserrno(unsigned long * _Value); 
#line 134 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\errno.h"
}__pragma( pack ( pop )) 
#line 136
#pragma warning(pop)
#line 12 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Enterprise\\VC\\Tools\\MSVC\\14.37.32822\\include\\vcruntime_string.h"
#pragma warning(push)
#pragma warning(disable: 4514 4820 )
#line 17
__pragma( pack ( push, 8 )) extern "C" {
#line 21
[[nodiscard]] const void *__cdecl 
#line 22
memchr(const void * _Buf, int _Val, size_t _MaxCount); 
#line 28
[[nodiscard]] int __cdecl 
#line 29
memcmp(const void * _Buf1, const void * _Buf2, size_t _Size); 
#line 43 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Enterprise\\VC\\Tools\\MSVC\\14.37.32822\\include\\vcruntime_string.h"
void *__cdecl memcpy(void * _Dst, const void * _Src, size_t _Size); 
#line 50
void *__cdecl memmove(void * _Dst, const void * _Src, size_t _Size); 
#line 63 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Enterprise\\VC\\Tools\\MSVC\\14.37.32822\\include\\vcruntime_string.h"
void *__cdecl memset(void * _Dst, int _Val, size_t _Size); 
#line 69
[[nodiscard]] const char *__cdecl 
#line 70
strchr(const char * _Str, int _Val); 
#line 75
[[nodiscard]] const char *__cdecl 
#line 76
strrchr(const char * _Str, int _Ch); 
#line 81
[[nodiscard]] const char *__cdecl 
#line 82
strstr(const char * _Str, const char * _SubStr); 
#line 87
[[nodiscard]] const __wchar_t *__cdecl 
#line 89
wcschr(const __wchar_t * _Str, __wchar_t _Ch); 
#line 94
[[nodiscard]] const __wchar_t *__cdecl 
#line 95
wcsrchr(const __wchar_t * _Str, __wchar_t _Ch); 
#line 100
[[nodiscard]] const __wchar_t *__cdecl 
#line 102
wcsstr(const __wchar_t * _Str, const __wchar_t * _SubStr); 
#line 109
}__pragma( pack ( pop )) 
#line 113 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Enterprise\\VC\\Tools\\MSVC\\14.37.32822\\include\\vcruntime_string.h"
#pragma warning(pop)
#line 14 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\corecrt_memcpy_s.h"
#pragma warning(push)
#pragma warning(disable: 4324 4514 4574 4710 4793 4820 4995 4996 28719 28726 28727 )
#line 18
__pragma( pack ( push, 8 )) extern "C" {
#line 39 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\corecrt_memcpy_s.h"
static __inline errno_t __cdecl memcpy_s(void *const 
#line 40
_Destination, const rsize_t 
#line 41
_DestinationSize, const void *const 
#line 42
_Source, const rsize_t 
#line 43
_SourceSize) 
#line 45
{ 
#line 46
if (_SourceSize == (0)) 
#line 47
{ 
#line 48
return 0; 
#line 49
}  
#line 51
{ int _Expr_val = !(!(_Destination != (0))); if (!_Expr_val) { (*_errno()) = 22; _invalid_parameter_noinfo(); return 22; }  } ; 
#line 52
if ((_Source == (0)) || (_DestinationSize < _SourceSize)) 
#line 53
{ 
#line 54
memset(_Destination, 0, _DestinationSize); 
#line 56
{ int _Expr_val = !(!(_Source != (0))); if (!_Expr_val) { (*_errno()) = 22; _invalid_parameter_noinfo(); return 22; }  } ; 
#line 57
{ int _Expr_val = !(!(_DestinationSize >= _SourceSize)); if (!_Expr_val) { (*_errno()) = 34; _invalid_parameter_noinfo(); return 34; }  } ; 
#line 60
return 22; 
#line 61
}  
#line 62
memcpy(_Destination, _Source, _SourceSize); 
#line 63
return 0; 
#line 64
} 
#line 67
static __inline errno_t __cdecl memmove_s(void *const 
#line 68
_Destination, const rsize_t 
#line 69
_DestinationSize, const void *const 
#line 70
_Source, const rsize_t 
#line 71
_SourceSize) 
#line 73
{ 
#line 74
if (_SourceSize == (0)) 
#line 75
{ 
#line 76
return 0; 
#line 77
}  
#line 79
{ int _Expr_val = !(!(_Destination != (0))); if (!_Expr_val) { (*_errno()) = 22; _invalid_parameter_noinfo(); return 22; }  } ; 
#line 80
{ int _Expr_val = !(!(_Source != (0))); if (!_Expr_val) { (*_errno()) = 22; _invalid_parameter_noinfo(); return 22; }  } ; 
#line 81
{ int _Expr_val = !(!(_DestinationSize >= _SourceSize)); if (!_Expr_val) { (*_errno()) = 34; _invalid_parameter_noinfo(); return 34; }  } ; 
#line 83
memmove(_Destination, _Source, _SourceSize); 
#line 84
return 0; 
#line 85
} 
#line 93 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\corecrt_memcpy_s.h"
}
#line 92
#pragma warning(pop)
__pragma( pack ( pop )) 
#line 17 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\corecrt_memory.h"
#pragma warning(push)
#pragma warning(disable: 4324 4514 4574 4710 4793 4820 4995 4996 28719 28726 28727 )
#line 23
__pragma( pack ( push, 8 )) extern "C" {
#line 28
int __cdecl _memicmp(const void * _Buf1, const void * _Buf2, size_t _Size); 
#line 35
int __cdecl _memicmp_l(const void * _Buf1, const void * _Buf2, size_t _Size, _locale_t _Locale); 
#line 83 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\corecrt_memory.h"
void *__cdecl memccpy(void * _Dst, const void * _Src, int _Val, size_t _Size); 
#line 91
int __cdecl memicmp(const void * _Buf1, const void * _Buf2, size_t _Size); 
#line 104 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\corecrt_memory.h"
extern "C++" inline void *__cdecl memchr(void *
#line 105
_Pv, int 
#line 106
_C, size_t 
#line 107
_N) 
#line 109
{ 
#line 110
const void *const _Pvc = _Pv; 
#line 111
return const_cast< void *>(memchr(_Pvc, _C, _N)); 
#line 112
} 
#line 118 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\corecrt_memory.h"
}__pragma( pack ( pop )) 
#line 122 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\corecrt_memory.h"
#pragma warning(pop)
#line 14 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\corecrt_wstring.h"
#pragma warning(push)
#pragma warning(disable: 4324 4514 4574 4710 4793 4820 4995 4996 28719 28726 28727 )
#line 20
__pragma( pack ( push, 8 )) extern "C" {
#line 32
errno_t __cdecl wcscat_s(__wchar_t * _Destination, rsize_t _SizeInWords, const __wchar_t * _Source); 
#line 39
errno_t __cdecl wcscpy_s(__wchar_t * _Destination, rsize_t _SizeInWords, const __wchar_t * _Source); 
#line 46
errno_t __cdecl wcsncat_s(__wchar_t * _Destination, rsize_t _SizeInWords, const __wchar_t * _Source, rsize_t _MaxCount); 
#line 54
errno_t __cdecl wcsncpy_s(__wchar_t * _Destination, rsize_t _SizeInWords, const __wchar_t * _Source, rsize_t _MaxCount); 
#line 62
__wchar_t *__cdecl wcstok_s(__wchar_t * _String, const __wchar_t * _Delimiter, __wchar_t ** _Context); 
#line 83 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\corecrt_wstring.h"
__declspec(allocator) __wchar_t *__cdecl _wcsdup(const __wchar_t * _String); 
#line 93 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\corecrt_wstring.h"
extern "C++" {template < size_t _Size > inline errno_t __cdecl wcscat_s ( wchar_t ( & _Destination ) [ _Size ], wchar_t const * _Source ) throw ( ) { return wcscat_s ( _Destination, _Size, _Source ); }}
#line 100 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\corecrt_wstring.h"
__wchar_t *__cdecl wcscat(__wchar_t * _Destination, const __wchar_t * _Source); 
#line 108 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\corecrt_wstring.h"
int __cdecl wcscmp(const __wchar_t * _String1, const __wchar_t * _String2); 
#line 113
extern "C++" {template < size_t _Size > inline errno_t __cdecl wcscpy_s ( wchar_t ( & _Destination ) [ _Size ], wchar_t const * _Source ) throw ( ) { return wcscpy_s ( _Destination, _Size, _Source ); }}
#line 119 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\corecrt_wstring.h"
__wchar_t *__cdecl wcscpy(__wchar_t * _Destination, const __wchar_t * _Source); 
#line 126 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\corecrt_wstring.h"
size_t __cdecl wcscspn(const __wchar_t * _String, const __wchar_t * _Control); 
#line 132
size_t __cdecl wcslen(const __wchar_t * _String); 
#line 145 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\corecrt_wstring.h"
size_t __cdecl wcsnlen(const __wchar_t * _Source, size_t _MaxCount); 
#line 161 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\corecrt_wstring.h"
static __inline size_t __cdecl wcsnlen_s(const __wchar_t *
#line 162
_Source, size_t 
#line 163
_MaxCount) 
#line 165
{ 
#line 166
return (_Source == (0)) ? 0 : wcsnlen(_Source, _MaxCount); 
#line 167
} 
#line 171 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\corecrt_wstring.h"
extern "C++" {template < size_t _Size > inline errno_t __cdecl wcsncat_s ( wchar_t ( & _Destination ) [ _Size ], wchar_t const * _Source, size_t _Count ) throw ( ) { return wcsncat_s ( _Destination, _Size, _Source, _Count ); }}
#line 178 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\corecrt_wstring.h"
__wchar_t *__cdecl wcsncat(__wchar_t * _Destination, const __wchar_t * _Source, size_t _Count); 
#line 187 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\corecrt_wstring.h"
int __cdecl wcsncmp(const __wchar_t * _String1, const __wchar_t * _String2, size_t _MaxCount); 
#line 193
extern "C++" {template < size_t _Size > inline errno_t __cdecl wcsncpy_s ( wchar_t ( & _Destination ) [ _Size ], wchar_t const * _Source, size_t _Count ) throw ( ) { return wcsncpy_s ( _Destination, _Size, _Source, _Count ); }}
#line 200 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\corecrt_wstring.h"
__wchar_t *__cdecl wcsncpy(__wchar_t * _Destination, const __wchar_t * _Source, size_t _Count); 
#line 209 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\corecrt_wstring.h"
const __wchar_t *__cdecl wcspbrk(const __wchar_t * _String, const __wchar_t * _Control); 
#line 215
size_t __cdecl wcsspn(const __wchar_t * _String, const __wchar_t * _Control); 
#line 221
__wchar_t *__cdecl wcstok(__wchar_t * _String, const __wchar_t * _Delimiter, __wchar_t ** _Context); 
#line 239 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\corecrt_wstring.h"
static __inline __wchar_t *__cdecl _wcstok(__wchar_t *const 
#line 240
_String, const __wchar_t *const 
#line 241
_Delimiter) 
#line 243
{ 
#line 244
return wcstok(_String, _Delimiter, 0); 
#line 245
} 
#line 253 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\corecrt_wstring.h"
extern "C++" 
#line 252
__declspec(deprecated("wcstok has been changed to conform with the ISO C standard, adding an extra context parameter. To use the legacy Microsoft wcsto" "k, define _CRT_NON_CONFORMING_WCSTOK.")) inline __wchar_t *__cdecl 
#line 253
wcstok(__wchar_t *
#line 254
_String, const __wchar_t *
#line 255
_Delimiter) throw() 
#line 257
{ 
#line 258
return wcstok(_String, _Delimiter, 0); 
#line 259
} 
#line 268 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\corecrt_wstring.h"
__wchar_t *__cdecl _wcserror(int _ErrorNumber); 
#line 273
errno_t __cdecl _wcserror_s(__wchar_t * _Buffer, size_t _SizeInWords, int _ErrorNumber); 
#line 279
extern "C++" {template < size_t _Size > inline errno_t __cdecl _wcserror_s ( wchar_t ( & _Buffer ) [ _Size ], int _Error ) throw ( ) { return _wcserror_s ( _Buffer, _Size, _Error ); }}
#line 288 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\corecrt_wstring.h"
__wchar_t *__cdecl __wcserror(const __wchar_t * _String); 
#line 292
errno_t __cdecl __wcserror_s(__wchar_t * _Buffer, size_t _SizeInWords, const __wchar_t * _ErrorMessage); 
#line 298
extern "C++" {template < size_t _Size > inline errno_t __cdecl __wcserror_s ( wchar_t ( & _Buffer ) [ _Size ], wchar_t const * _ErrorMessage ) throw ( ) { return __wcserror_s ( _Buffer, _Size, _ErrorMessage ); }}
#line 304 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\corecrt_wstring.h"
int __cdecl _wcsicmp(const __wchar_t * _String1, const __wchar_t * _String2); 
#line 309
int __cdecl _wcsicmp_l(const __wchar_t * _String1, const __wchar_t * _String2, _locale_t _Locale); 
#line 315
int __cdecl _wcsnicmp(const __wchar_t * _String1, const __wchar_t * _String2, size_t _MaxCount); 
#line 321
int __cdecl _wcsnicmp_l(const __wchar_t * _String1, const __wchar_t * _String2, size_t _MaxCount, _locale_t _Locale); 
#line 328
errno_t __cdecl _wcsnset_s(__wchar_t * _Destination, size_t _SizeInWords, __wchar_t _Value, size_t _MaxCount); 
#line 335
extern "C++" {template < size_t _Size > inline errno_t __cdecl _wcsnset_s ( wchar_t ( & _Destination ) [ _Size ], wchar_t _Value, size_t _MaxCount ) throw ( ) { return _wcsnset_s ( _Destination, _Size, _Value, _MaxCount ); }}
#line 342 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\corecrt_wstring.h"
__wchar_t *__cdecl _wcsnset(__wchar_t * _String, __wchar_t _Value, size_t _MaxCount); 
#line 350 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\corecrt_wstring.h"
__wchar_t *__cdecl _wcsrev(__wchar_t * _String); 
#line 354
errno_t __cdecl _wcsset_s(__wchar_t * _Destination, size_t _SizeInWords, __wchar_t _Value); 
#line 360
extern "C++" {template < size_t _Size > inline errno_t __cdecl _wcsset_s ( wchar_t ( & _String ) [ _Size ], wchar_t _Value ) throw ( ) { return _wcsset_s ( _String, _Size, _Value ); }}
#line 366 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\corecrt_wstring.h"
__wchar_t *__cdecl _wcsset(__wchar_t * _String, __wchar_t _Value); 
#line 373 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\corecrt_wstring.h"
errno_t __cdecl _wcslwr_s(__wchar_t * _String, size_t _SizeInWords); 
#line 378
extern "C++" {template < size_t _Size > inline errno_t __cdecl _wcslwr_s ( wchar_t ( & _String ) [ _Size ] ) throw ( ) { return _wcslwr_s ( _String, _Size ); }}
#line 383 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\corecrt_wstring.h"
__wchar_t *__cdecl _wcslwr(__wchar_t * _String); 
#line 389 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\corecrt_wstring.h"
errno_t __cdecl _wcslwr_s_l(__wchar_t * _String, size_t _SizeInWords, _locale_t _Locale); 
#line 395
extern "C++" {template < size_t _Size > inline errno_t __cdecl _wcslwr_s_l ( wchar_t ( & _String ) [ _Size ], _locale_t _Locale ) throw ( ) { return _wcslwr_s_l ( _String, _Size, _Locale ); }}
#line 401 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\corecrt_wstring.h"
__wchar_t *__cdecl _wcslwr_l(__wchar_t * _String, _locale_t _Locale); 
#line 409 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\corecrt_wstring.h"
errno_t __cdecl _wcsupr_s(__wchar_t * _String, size_t _Size); 
#line 414
extern "C++" {template < size_t _Size > inline errno_t __cdecl _wcsupr_s ( wchar_t ( & _String ) [ _Size ] ) throw ( ) { return _wcsupr_s ( _String, _Size ); }}
#line 419 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\corecrt_wstring.h"
__wchar_t *__cdecl _wcsupr(__wchar_t * _String); 
#line 425 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\corecrt_wstring.h"
errno_t __cdecl _wcsupr_s_l(__wchar_t * _String, size_t _Size, _locale_t _Locale); 
#line 431
extern "C++" {template < size_t _Size > inline errno_t __cdecl _wcsupr_s_l ( wchar_t ( & _String ) [ _Size ], _locale_t _Locale ) throw ( ) { return _wcsupr_s_l ( _String, _Size, _Locale ); }}
#line 437 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\corecrt_wstring.h"
__wchar_t *__cdecl _wcsupr_l(__wchar_t * _String, _locale_t _Locale); 
#line 446 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\corecrt_wstring.h"
size_t __cdecl wcsxfrm(__wchar_t * _Destination, const __wchar_t * _Source, size_t _MaxCount); 
#line 454
size_t __cdecl _wcsxfrm_l(__wchar_t * _Destination, const __wchar_t * _Source, size_t _MaxCount, _locale_t _Locale); 
#line 462
int __cdecl wcscoll(const __wchar_t * _String1, const __wchar_t * _String2); 
#line 468
int __cdecl _wcscoll_l(const __wchar_t * _String1, const __wchar_t * _String2, _locale_t _Locale); 
#line 475
int __cdecl _wcsicoll(const __wchar_t * _String1, const __wchar_t * _String2); 
#line 481
int __cdecl _wcsicoll_l(const __wchar_t * _String1, const __wchar_t * _String2, _locale_t _Locale); 
#line 488
int __cdecl _wcsncoll(const __wchar_t * _String1, const __wchar_t * _String2, size_t _MaxCount); 
#line 495
int __cdecl _wcsncoll_l(const __wchar_t * _String1, const __wchar_t * _String2, size_t _MaxCount, _locale_t _Locale); 
#line 503
int __cdecl _wcsnicoll(const __wchar_t * _String1, const __wchar_t * _String2, size_t _MaxCount); 
#line 510
int __cdecl _wcsnicoll_l(const __wchar_t * _String1, const __wchar_t * _String2, size_t _MaxCount, _locale_t _Locale); 
#line 525
extern "C++" {
#line 529
inline __wchar_t *__cdecl wcschr(__wchar_t *_String, __wchar_t _C) 
#line 530
{ 
#line 531
return const_cast< __wchar_t *>(wcschr(static_cast< const __wchar_t *>(_String), _C)); 
#line 532
} 
#line 535
inline __wchar_t *__cdecl wcspbrk(__wchar_t *_String, const __wchar_t *_Control) 
#line 536
{ 
#line 537
return const_cast< __wchar_t *>(wcspbrk(static_cast< const __wchar_t *>(_String), _Control)); 
#line 538
} 
#line 541
inline __wchar_t *__cdecl wcsrchr(__wchar_t *_String, __wchar_t _C) 
#line 542
{ 
#line 543
return const_cast< __wchar_t *>(wcsrchr(static_cast< const __wchar_t *>(_String), _C)); 
#line 544
} 
#line 548
inline __wchar_t *__cdecl wcsstr(__wchar_t *_String, const __wchar_t *_SubStr) 
#line 549
{ 
#line 550
return const_cast< __wchar_t *>(wcsstr(static_cast< const __wchar_t *>(_String), _SubStr)); 
#line 551
} 
#line 553
}
#line 570 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\corecrt_wstring.h"
__wchar_t *__cdecl wcsdup(const __wchar_t * _String); 
#line 582 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\corecrt_wstring.h"
int __cdecl wcsicmp(const __wchar_t * _String1, const __wchar_t * _String2); 
#line 588
int __cdecl wcsnicmp(const __wchar_t * _String1, const __wchar_t * _String2, size_t _MaxCount); 
#line 596
__wchar_t *__cdecl wcsnset(__wchar_t * _String, __wchar_t _Value, size_t _MaxCount); 
#line 604
__wchar_t *__cdecl wcsrev(__wchar_t * _String); 
#line 610
__wchar_t *__cdecl wcsset(__wchar_t * _String, __wchar_t _Value); 
#line 617
__wchar_t *__cdecl wcslwr(__wchar_t * _String); 
#line 623
__wchar_t *__cdecl wcsupr(__wchar_t * _String); 
#line 628
int __cdecl wcsicoll(const __wchar_t * _String1, const __wchar_t * _String2); 
#line 637 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\corecrt_wstring.h"
}__pragma( pack ( pop )) 
#line 641 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\corecrt_wstring.h"
#pragma warning(pop)
#line 19 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\string.h"
#pragma warning(push)
#pragma warning(disable: 4324 4514 4574 4710 4793 4820 4995 4996 28719 28726 28727 )
#line 23
__pragma( pack ( push, 8 )) extern "C" {
#line 32
errno_t __cdecl strcpy_s(char * _Destination, rsize_t _SizeInBytes, const char * _Source); 
#line 39
errno_t __cdecl strcat_s(char * _Destination, rsize_t _SizeInBytes, const char * _Source); 
#line 46
errno_t __cdecl strerror_s(char * _Buffer, size_t _SizeInBytes, int _ErrorNumber); 
#line 52
errno_t __cdecl strncat_s(char * _Destination, rsize_t _SizeInBytes, const char * _Source, rsize_t _MaxCount); 
#line 60
errno_t __cdecl strncpy_s(char * _Destination, rsize_t _SizeInBytes, const char * _Source, rsize_t _MaxCount); 
#line 68
char *__cdecl strtok_s(char * _String, const char * _Delimiter, char ** _Context); 
#line 76 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\string.h"
void *__cdecl _memccpy(void * _Dst, const void * _Src, int _Val, size_t _MaxCount); 
#line 83
extern "C++" {template < size_t _Size > inline errno_t __cdecl strcat_s ( char ( & _Destination ) [ _Size ], char const * _Source ) throw ( ) { return strcat_s ( _Destination, _Size, _Source ); }}
#line 91 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\string.h"
char *__cdecl strcat(char * _Destination, const char * _Source); 
#line 100 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\string.h"
int __cdecl strcmp(const char * _Str1, const char * _Str2); 
#line 106
int __cdecl _strcmpi(const char * _String1, const char * _String2); 
#line 112
int __cdecl strcoll(const char * _String1, const char * _String2); 
#line 118
int __cdecl _strcoll_l(const char * _String1, const char * _String2, _locale_t _Locale); 
#line 124
extern "C++" {template < size_t _Size > inline errno_t __cdecl strcpy_s ( char ( & _Destination ) [ _Size ], char const * _Source ) throw ( ) { return strcpy_s ( _Destination, _Size, _Source ); }}
#line 130 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\string.h"
char *__cdecl strcpy(char * _Destination, const char * _Source); 
#line 137 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\string.h"
size_t __cdecl strcspn(const char * _Str, const char * _Control); 
#line 148 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\string.h"
__declspec(allocator) char *__cdecl _strdup(const char * _Source); 
#line 159 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\string.h"
char *__cdecl _strerror(const char * _ErrorMessage); 
#line 164
errno_t __cdecl _strerror_s(char * _Buffer, size_t _SizeInBytes, const char * _ErrorMessage); 
#line 170
extern "C++" {template < size_t _Size > inline errno_t __cdecl _strerror_s ( char ( & _Buffer ) [ _Size ], char const * _ErrorMessage ) throw ( ) { return _strerror_s ( _Buffer, _Size, _ErrorMessage ); }}
#line 178 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\string.h"
char *__cdecl strerror(int _ErrorMessage); 
#line 182
extern "C++" {template < size_t _Size > inline errno_t __cdecl strerror_s ( char ( & _Buffer ) [ _Size ], int _ErrorMessage ) throw ( ) { return strerror_s ( _Buffer, _Size, _ErrorMessage ); }}
#line 189 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\string.h"
int __cdecl _stricmp(const char * _String1, const char * _String2); 
#line 195
int __cdecl _stricoll(const char * _String1, const char * _String2); 
#line 201
int __cdecl _stricoll_l(const char * _String1, const char * _String2, _locale_t _Locale); 
#line 208
int __cdecl _stricmp_l(const char * _String1, const char * _String2, _locale_t _Locale); 
#line 215
size_t __cdecl strlen(const char * _Str); 
#line 220
errno_t __cdecl _strlwr_s(char * _String, size_t _Size); 
#line 225
extern "C++" {template < size_t _Size > inline errno_t __cdecl _strlwr_s ( char ( & _String ) [ _Size ] ) throw ( ) { return _strlwr_s ( _String, _Size ); }}
#line 230 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\string.h"
char *__cdecl _strlwr(char * _String); 
#line 236 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\string.h"
errno_t __cdecl _strlwr_s_l(char * _String, size_t _Size, _locale_t _Locale); 
#line 242
extern "C++" {template < size_t _Size > inline errno_t __cdecl _strlwr_s_l ( char ( & _String ) [ _Size ], _locale_t _Locale ) throw ( ) { return _strlwr_s_l ( _String, _Size, _Locale ); }}
#line 248 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\string.h"
char *__cdecl _strlwr_l(char * _String, _locale_t _Locale); 
#line 255 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\string.h"
extern "C++" {template < size_t _Size > inline errno_t __cdecl strncat_s ( char ( & _Destination ) [ _Size ], char const * _Source, size_t _Count ) throw ( ) { return strncat_s ( _Destination, _Size, _Source, _Count ); }}
#line 262 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\string.h"
char *__cdecl strncat(char * _Destination, const char * _Source, size_t _Count); 
#line 271 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\string.h"
int __cdecl strncmp(const char * _Str1, const char * _Str2, size_t _MaxCount); 
#line 278
int __cdecl _strnicmp(const char * _String1, const char * _String2, size_t _MaxCount); 
#line 285
int __cdecl _strnicmp_l(const char * _String1, const char * _String2, size_t _MaxCount, _locale_t _Locale); 
#line 293
int __cdecl _strnicoll(const char * _String1, const char * _String2, size_t _MaxCount); 
#line 300
int __cdecl _strnicoll_l(const char * _String1, const char * _String2, size_t _MaxCount, _locale_t _Locale); 
#line 308
int __cdecl _strncoll(const char * _String1, const char * _String2, size_t _MaxCount); 
#line 315
int __cdecl _strncoll_l(const char * _String1, const char * _String2, size_t _MaxCount, _locale_t _Locale); 
#line 322
size_t __cdecl __strncnt(const char * _String, size_t _Count); 
#line 327
extern "C++" {template < size_t _Size > inline errno_t __cdecl strncpy_s ( char ( & _Destination ) [ _Size ], char const * _Source, size_t _Count ) throw ( ) { return strncpy_s ( _Destination, _Size, _Source, _Count ); }}
#line 334 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\string.h"
char *__cdecl strncpy(char * _Destination, const char * _Source, size_t _Count); 
#line 351 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\string.h"
size_t __cdecl strnlen(const char * _String, size_t _MaxCount); 
#line 367 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\string.h"
static __inline size_t __cdecl strnlen_s(const char *
#line 368
_String, size_t 
#line 369
_MaxCount) 
#line 371
{ 
#line 372
return (_String == (0)) ? 0 : strnlen(_String, _MaxCount); 
#line 373
} 
#line 378 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\string.h"
errno_t __cdecl _strnset_s(char * _String, size_t _SizeInBytes, int _Value, size_t _MaxCount); 
#line 385
extern "C++" {template < size_t _Size > inline errno_t __cdecl _strnset_s ( char ( & _Destination ) [ _Size ], int _Value, size_t _Count ) throw ( ) { return _strnset_s ( _Destination, _Size, _Value, _Count ); }}
#line 392 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\string.h"
char *__cdecl _strnset(char * _Destination, int _Value, size_t _Count); 
#line 401 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\string.h"
const char *__cdecl strpbrk(const char * _Str, const char * _Control); 
#line 406
char *__cdecl _strrev(char * _Str); 
#line 411
errno_t __cdecl _strset_s(char * _Destination, size_t _DestinationSize, int _Value); 
#line 417
extern "C++" {template < size_t _Size > inline errno_t __cdecl _strset_s ( char ( & _Destination ) [ _Size ], int _Value ) throw ( ) { return _strset_s ( _Destination, _Size, _Value ); }}
#line 423 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\string.h"
char *__cdecl _strset(char * _Destination, int _Value); 
#line 430 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\string.h"
size_t __cdecl strspn(const char * _Str, const char * _Control); 
#line 436
char *__cdecl strtok(char * _String, const char * _Delimiter); 
#line 442
errno_t __cdecl _strupr_s(char * _String, size_t _Size); 
#line 447
extern "C++" {template < size_t _Size > inline errno_t __cdecl _strupr_s ( char ( & _String ) [ _Size ] ) throw ( ) { return _strupr_s ( _String, _Size ); }}
#line 452 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\string.h"
char *__cdecl _strupr(char * _String); 
#line 458 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\string.h"
errno_t __cdecl _strupr_s_l(char * _String, size_t _Size, _locale_t _Locale); 
#line 464
extern "C++" {template < size_t _Size > inline errno_t __cdecl _strupr_s_l ( char ( & _String ) [ _Size ], _locale_t _Locale ) throw ( ) { return _strupr_s_l ( _String, _Size, _Locale ); }}
#line 470 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\string.h"
char *__cdecl _strupr_l(char * _String, _locale_t _Locale); 
#line 479 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\string.h"
size_t __cdecl strxfrm(char * _Destination, const char * _Source, size_t _MaxCount); 
#line 487
size_t __cdecl _strxfrm_l(char * _Destination, const char * _Source, size_t _MaxCount, _locale_t _Locale); 
#line 497
extern "C++" {
#line 500
inline char *__cdecl strchr(char *const _String, const int _Ch) 
#line 501
{ 
#line 502
return const_cast< char *>(strchr(static_cast< const char *>(_String), _Ch)); 
#line 503
} 
#line 506
inline char *__cdecl strpbrk(char *const _String, const char *const _Control) 
#line 507
{ 
#line 508
return const_cast< char *>(strpbrk(static_cast< const char *>(_String), _Control)); 
#line 509
} 
#line 512
inline char *__cdecl strrchr(char *const _String, const int _Ch) 
#line 513
{ 
#line 514
return const_cast< char *>(strrchr(static_cast< const char *>(_String), _Ch)); 
#line 515
} 
#line 518
inline char *__cdecl strstr(char *const _String, const char *const _SubString) 
#line 519
{ 
#line 520
return const_cast< char *>(strstr(static_cast< const char *>(_String), _SubString)); 
#line 521
} 
#line 522
}
#line 532 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\string.h"
char *__cdecl strdup(const char * _String); 
#line 539
int __cdecl strcmpi(const char * _String1, const char * _String2); 
#line 545
int __cdecl stricmp(const char * _String1, const char * _String2); 
#line 551
char *__cdecl strlwr(char * _String); 
#line 556
int __cdecl strnicmp(const char * _String1, const char * _String2, size_t _MaxCount); 
#line 563
char *__cdecl strnset(char * _String, int _Value, size_t _MaxCount); 
#line 570
char *__cdecl strrev(char * _String); 
#line 575
char *__cdecl strset(char * _String, int _Value); 
#line 580
char *__cdecl strupr(char * _String); 
#line 588 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\string.h"
}__pragma( pack ( pop )) 
#line 590
#pragma warning(pop)
#line 13 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\corecrt_wtime.h"
#pragma warning(push)
#pragma warning(disable: 4324 4514 4574 4710 4793 4820 4995 4996 28719 28726 28727 )
#line 17
__pragma( pack ( push, 8 )) extern "C" {
#line 26
struct tm { 
#line 28
int tm_sec; 
#line 29
int tm_min; 
#line 30
int tm_hour; 
#line 31
int tm_mday; 
#line 32
int tm_mon; 
#line 33
int tm_year; 
#line 34
int tm_wday; 
#line 35
int tm_yday; 
#line 36
int tm_isdst; 
#line 37
}; 
#line 48
__wchar_t *__cdecl _wasctime(const tm * _Tm); 
#line 54
errno_t __cdecl _wasctime_s(__wchar_t * _Buffer, size_t _SizeInWords, const tm * _Tm); 
#line 60
extern "C++" {template < size_t _Size > inline errno_t __cdecl _wasctime_s ( wchar_t ( & _Buffer ) [ _Size ], struct tm const * _Time ) throw ( ) { return _wasctime_s ( _Buffer, _Size, _Time ); }}
#line 69 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\corecrt_wtime.h"
size_t __cdecl wcsftime(__wchar_t * _Buffer, size_t _SizeInWords, const __wchar_t * _Format, const tm * _Tm); 
#line 78
size_t __cdecl _wcsftime_l(__wchar_t * _Buffer, size_t _SizeInWords, const __wchar_t * _Format, const tm * _Tm, _locale_t _Locale); 
#line 88
__wchar_t *__cdecl _wctime32(const __time32_t * _Time); 
#line 93
errno_t __cdecl _wctime32_s(__wchar_t * _Buffer, size_t _SizeInWords, const __time32_t * _Time); 
#line 99
extern "C++" {template < size_t _Size > inline errno_t __cdecl _wctime32_s ( wchar_t ( & _Buffer ) [ _Size ], __time32_t const * _Time ) throw ( ) { return _wctime32_s ( _Buffer, _Size, _Time ); }}
#line 108 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\corecrt_wtime.h"
__wchar_t *__cdecl _wctime64(const __time64_t * _Time); 
#line 113
errno_t __cdecl _wctime64_s(__wchar_t * _Buffer, size_t _SizeInWords, const __time64_t * _Time); 
#line 118
extern "C++" {template < size_t _Size > inline errno_t __cdecl _wctime64_s ( wchar_t ( & _Buffer ) [ _Size ], __time64_t const * _Time ) throw ( ) { return _wctime64_s ( _Buffer, _Size, _Time ); }}
#line 125 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\corecrt_wtime.h"
errno_t __cdecl _wstrdate_s(__wchar_t * _Buffer, size_t _SizeInWords); 
#line 130
extern "C++" {template < size_t _Size > inline errno_t __cdecl _wstrdate_s ( wchar_t ( & _Buffer ) [ _Size ] ) throw ( ) { return _wstrdate_s ( _Buffer, _Size ); }}
#line 135 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\corecrt_wtime.h"
__wchar_t *__cdecl _wstrdate(__wchar_t * _Buffer); 
#line 141 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\corecrt_wtime.h"
errno_t __cdecl _wstrtime_s(__wchar_t * _Buffer, size_t _SizeInWords); 
#line 146
extern "C++" {template < size_t _Size > inline errno_t __cdecl _wstrtime_s ( wchar_t ( & _Buffer ) [ _Size ] ) throw ( ) { return _wstrtime_s ( _Buffer, _Size ); }}
#line 151 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\corecrt_wtime.h"
__wchar_t *__cdecl _wstrtime(__wchar_t * _Buffer); 
#line 186 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\corecrt_wtime.h"
static __inline __wchar_t *__cdecl _wctime(const time_t *const 
#line 187
_Time) 
#line 188
{ 
#line 189
return _wctime64(_Time); 
#line 190
} 
#line 193
static __inline errno_t __cdecl _wctime_s(__wchar_t *const 
#line 194
_Buffer, const size_t 
#line 195
_SizeInWords, const time_t *const 
#line 196
_Time) 
#line 198
{ 
#line 199
return _wctime64_s(_Buffer, _SizeInWords, _Time); 
#line 200
} 
#line 205 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\corecrt_wtime.h"
}__pragma( pack ( pop )) 
#line 207
#pragma warning(pop)
#line 15 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\time.h"
#pragma warning(push)
#pragma warning(disable: 4324 4514 4574 4710 4793 4820 4995 4996 28719 28726 28727 )
#line 19
__pragma( pack ( push, 8 )) extern "C" {
#line 28
typedef long clock_t; 
#line 30
struct _timespec32 { 
#line 32
__time32_t tv_sec; 
#line 33
long tv_nsec; 
#line 34
}; 
#line 36
struct _timespec64 { 
#line 38
__time64_t tv_sec; 
#line 39
long tv_nsec; 
#line 40
}; 
#line 43
struct timespec { 
#line 45
time_t tv_sec; 
#line 46
long tv_nsec; 
#line 47
}; 
#line 66 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\time.h"
int *__cdecl __daylight(); 
#line 72
long *__cdecl __dstbias(); 
#line 78
long *__cdecl __timezone(); 
#line 84
char **__cdecl __tzname(); 
#line 89
errno_t __cdecl _get_daylight(int * _Daylight); 
#line 94
errno_t __cdecl _get_dstbias(long * _DaylightSavingsBias); 
#line 99
errno_t __cdecl _get_timezone(long * _TimeZone); 
#line 104
errno_t __cdecl _get_tzname(size_t * _ReturnValue, char * _Buffer, size_t _SizeInBytes, int _Index); 
#line 121
char *__cdecl asctime(const tm * _Tm); 
#line 128
errno_t __cdecl asctime_s(char * _Buffer, size_t _SizeInBytes, const tm * _Tm); 
#line 135 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\time.h"
extern "C++" {template < size_t _Size > inline errno_t __cdecl asctime_s ( char ( & _Buffer ) [ _Size ], struct tm const * _Time ) throw ( ) { return asctime_s ( _Buffer, _Size, _Time ); }}
#line 142 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\time.h"
clock_t __cdecl clock(); 
#line 147
char *__cdecl _ctime32(const __time32_t * _Time); 
#line 152
errno_t __cdecl _ctime32_s(char * _Buffer, size_t _SizeInBytes, const __time32_t * _Time); 
#line 158
extern "C++" {template < size_t _Size > inline errno_t __cdecl _ctime32_s ( char ( & _Buffer ) [ _Size ], __time32_t const * _Time ) throw ( ) { return _ctime32_s ( _Buffer, _Size, _Time ); }}
#line 167 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\time.h"
char *__cdecl _ctime64(const __time64_t * _Time); 
#line 172
errno_t __cdecl _ctime64_s(char * _Buffer, size_t _SizeInBytes, const __time64_t * _Time); 
#line 178
extern "C++" {template < size_t _Size > inline errno_t __cdecl _ctime64_s ( char ( & _Buffer ) [ _Size ], __time64_t const * _Time ) throw ( ) { return _ctime64_s ( _Buffer, _Size, _Time ); }}
#line 185 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\time.h"
double __cdecl _difftime32(__time32_t _Time1, __time32_t _Time2); 
#line 191
double __cdecl _difftime64(__time64_t _Time1, __time64_t _Time2); 
#line 198
tm *__cdecl _gmtime32(const __time32_t * _Time); 
#line 203
errno_t __cdecl _gmtime32_s(tm * _Tm, const __time32_t * _Time); 
#line 210
tm *__cdecl _gmtime64(const __time64_t * _Time); 
#line 215
errno_t __cdecl _gmtime64_s(tm * _Tm, const __time64_t * _Time); 
#line 222
tm *__cdecl _localtime32(const __time32_t * _Time); 
#line 227
errno_t __cdecl _localtime32_s(tm * _Tm, const __time32_t * _Time); 
#line 234
tm *__cdecl _localtime64(const __time64_t * _Time); 
#line 239
errno_t __cdecl _localtime64_s(tm * _Tm, const __time64_t * _Time); 
#line 245
__time32_t __cdecl _mkgmtime32(tm * _Tm); 
#line 250
__time64_t __cdecl _mkgmtime64(tm * _Tm); 
#line 255
__time32_t __cdecl _mktime32(tm * _Tm); 
#line 260
__time64_t __cdecl _mktime64(tm * _Tm); 
#line 266
size_t __cdecl strftime(char * _Buffer, size_t _SizeInBytes, const char * _Format, const tm * _Tm); 
#line 275
size_t __cdecl _strftime_l(char * _Buffer, size_t _MaxSize, const char * _Format, const tm * _Tm, _locale_t _Locale); 
#line 284
errno_t __cdecl _strdate_s(char * _Buffer, size_t _SizeInBytes); 
#line 289
extern "C++" {template < size_t _Size > inline errno_t __cdecl _strdate_s ( char ( & _Buffer ) [ _Size ] ) throw ( ) { return _strdate_s ( _Buffer, _Size ); }}
#line 294 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\time.h"
char *__cdecl _strdate(char * _Buffer); 
#line 300 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\time.h"
errno_t __cdecl _strtime_s(char * _Buffer, size_t _SizeInBytes); 
#line 305
extern "C++" {template < size_t _Size > inline errno_t __cdecl _strtime_s ( char ( & _Buffer ) [ _Size ] ) throw ( ) { return _strtime_s ( _Buffer, _Size ); }}
#line 310 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\time.h"
char *__cdecl _strtime(char * _Buffer); 
#line 315 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\time.h"
__time32_t __cdecl _time32(__time32_t * _Time); 
#line 319
__time64_t __cdecl _time64(__time64_t * _Time); 
#line 325
int __cdecl _timespec32_get(_timespec32 * _Ts, int _Base); 
#line 332
int __cdecl _timespec64_get(_timespec64 * _Ts, int _Base); 
#line 346
void __cdecl _tzset(); 
#line 349
__declspec(deprecated("This function or variable has been superceded by newer library or operating system functionality. Consider using GetLocalTime in" "stead. See online help for details.")) unsigned __cdecl 
#line 350
_getsystime(tm * _Tm); 
#line 354
__declspec(deprecated("This function or variable has been superceded by newer library or operating system functionality. Consider using SetLocalTime in" "stead. See online help for details.")) unsigned __cdecl 
#line 355
_setsystime(tm * _Tm, unsigned _Milliseconds); 
#line 471 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\time.h"
static __inline char *__cdecl ctime(const time_t *const 
#line 472
_Time) 
#line 474
{ 
#line 475
return _ctime64(_Time); 
#line 476
} 
#line 479
static __inline double __cdecl difftime(const time_t 
#line 480
_Time1, const time_t 
#line 481
_Time2) 
#line 483
{ 
#line 484
return _difftime64(_Time1, _Time2); 
#line 485
} 
#line 488
static __inline tm *__cdecl gmtime(const time_t *const 
#line 489
_Time) 
#line 490
{ 
#line 491
return _gmtime64(_Time); 
#line 492
} 
#line 495
static __inline tm *__cdecl localtime(const time_t *const 
#line 496
_Time) 
#line 498
{ 
#line 499
return _localtime64(_Time); 
#line 500
} 
#line 503
static __inline time_t __cdecl _mkgmtime(tm *const 
#line 504
_Tm) 
#line 506
{ 
#line 507
return _mkgmtime64(_Tm); 
#line 508
} 
#line 511
static __inline time_t __cdecl mktime(tm *const 
#line 512
_Tm) 
#line 514
{ 
#line 515
return _mktime64(_Tm); 
#line 516
} 
#line 518
static __inline time_t __cdecl time(time_t *const 
#line 519
_Time) 
#line 521
{ 
#line 522
return _time64(_Time); 
#line 523
} 
#line 526
static __inline int __cdecl timespec_get(timespec *const 
#line 527
_Ts, const int 
#line 528
_Base) 
#line 530
{ 
#line 531
return _timespec64_get((_timespec64 *)_Ts, _Base); 
#line 532
} 
#line 536
static __inline errno_t __cdecl ctime_s(char *const 
#line 537
_Buffer, const size_t 
#line 538
_SizeInBytes, const time_t *const 
#line 539
_Time) 
#line 541
{ 
#line 542
return _ctime64_s(_Buffer, _SizeInBytes, _Time); 
#line 543
} 
#line 546
static __inline errno_t __cdecl gmtime_s(tm *const 
#line 547
_Tm, const time_t *const 
#line 548
_Time) 
#line 550
{ 
#line 551
return _gmtime64_s(_Tm, _Time); 
#line 552
} 
#line 555
static __inline errno_t __cdecl localtime_s(tm *const 
#line 556
_Tm, const time_t *const 
#line 557
_Time) 
#line 559
{ 
#line 560
return _localtime64_s(_Tm, _Time); 
#line 561
} 
#line 580 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\time.h"
void __cdecl tzset(); 
#line 587 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\time.h"
}__pragma( pack ( pop )) 
#line 589
#pragma warning(pop)
#line 88 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt/common_functions.h"
extern "C" {
#line 91 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt/common_functions.h"
extern clock_t __cdecl clock(); 
#line 96 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt/common_functions.h"
extern void *__cdecl memset(void *, int, size_t); 
#line 97
extern void *__cdecl memcpy(void *, const void *, size_t); 
#line 99
}
#line 121 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.h"
extern "C" {
#line 219 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.h"
extern int __cdecl abs(int a); 
#line 227
extern long __cdecl labs(long a); 
#line 235
extern __int64 llabs(__int64 a); 
#line 285 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.h"
extern double __cdecl fabs(double x); 
#line 328
extern __inline float fabsf(float x); 
#line 338 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.h"
extern inline int min(const int a, const int b); 
#line 345
extern inline unsigned umin(const unsigned a, const unsigned b); 
#line 352
extern inline __int64 llmin(const __int64 a, const __int64 b); 
#line 359
extern inline unsigned __int64 ullmin(const unsigned __int64 a, const unsigned __int64 b); 
#line 382 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.h"
extern float __cdecl fminf(float x, float y); 
#line 402 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.h"
extern double __cdecl fmin(double x, double y); 
#line 413 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.h"
extern inline int max(const int a, const int b); 
#line 421
extern inline unsigned umax(const unsigned a, const unsigned b); 
#line 428
extern inline __int64 llmax(const __int64 a, const __int64 b); 
#line 435
extern inline unsigned __int64 ullmax(const unsigned __int64 a, const unsigned __int64 b); 
#line 458 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.h"
extern float __cdecl fmaxf(float x, float y); 
#line 478 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.h"
extern double __cdecl fmax(double, double); 
#line 520 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.h"
extern double __cdecl sin(double x); 
#line 553
extern double __cdecl cos(double x); 
#line 572 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.h"
extern void sincos(double x, double * sptr, double * cptr); 
#line 588
extern void sincosf(float x, float * sptr, float * cptr); 
#line 633 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.h"
extern double __cdecl tan(double x); 
#line 702
extern double __cdecl sqrt(double x); 
#line 774 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.h"
extern double rsqrt(double x); 
#line 844
extern float rsqrtf(float x); 
#line 902 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.h"
extern double __cdecl log2(double x); 
#line 967 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.h"
extern double __cdecl exp2(double x); 
#line 1032 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.h"
extern float __cdecl exp2f(float x); 
#line 1097 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.h"
extern double exp10(double x); 
#line 1160
extern float exp10f(float x); 
#line 1255 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.h"
extern double __cdecl expm1(double x); 
#line 1347 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.h"
extern float __cdecl expm1f(float x); 
#line 1403 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.h"
extern float __cdecl log2f(float x); 
#line 1455 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.h"
extern double __cdecl log10(double x); 
#line 1525
extern double __cdecl log(double x); 
#line 1623 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.h"
extern double __cdecl log1p(double x); 
#line 1722 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.h"
extern float __cdecl log1pf(float x); 
#line 1784 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.h"
extern double __cdecl floor(double x); 
#line 1863
extern double __cdecl exp(double x); 
#line 1904
extern double __cdecl cosh(double x); 
#line 1954
extern double __cdecl sinh(double x); 
#line 2004
extern double __cdecl tanh(double x); 
#line 2061 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.h"
extern double __cdecl acosh(double x); 
#line 2119 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.h"
extern float __cdecl acoshf(float x); 
#line 2172 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.h"
extern double __cdecl asinh(double x); 
#line 2225 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.h"
extern float __cdecl asinhf(float x); 
#line 2279 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.h"
extern double __cdecl atanh(double x); 
#line 2333 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.h"
extern float __cdecl atanhf(float x); 
#line 2380 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.h"
extern double __cdecl ldexp(double x, int exp); 
#line 2426
extern __inline float ldexpf(float x, int exp); 
#line 2480 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.h"
extern double __cdecl logb(double x); 
#line 2535 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.h"
extern float __cdecl logbf(float x); 
#line 2575 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.h"
extern int __cdecl ilogb(double x); 
#line 2615 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.h"
extern int __cdecl ilogbf(float x); 
#line 2691 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.h"
extern double __cdecl scalbn(double x, int n); 
#line 2767 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.h"
extern float __cdecl scalbnf(float x, int n); 
#line 2843 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.h"
extern double __cdecl scalbln(double x, long n); 
#line 2919 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.h"
extern float __cdecl scalblnf(float x, long n); 
#line 2994 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.h"
extern double __cdecl frexp(double x, int * nptr); 
#line 3068
extern __inline float frexpf(float x, int * nptr); 
#line 3122 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.h"
extern double __cdecl round(double x); 
#line 3177 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.h"
extern float __cdecl roundf(float x); 
#line 3195 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.h"
extern long __cdecl lround(double x); 
#line 3213 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.h"
extern long __cdecl lroundf(float x); 
#line 3231 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.h"
extern __int64 __cdecl llround(double x); 
#line 3249 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.h"
extern __int64 __cdecl llroundf(float x); 
#line 3377 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.h"
extern float __cdecl rintf(float x); 
#line 3394 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.h"
extern long __cdecl lrint(double x); 
#line 3411 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.h"
extern long __cdecl lrintf(float x); 
#line 3428 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.h"
extern __int64 __cdecl llrint(double x); 
#line 3445 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.h"
extern __int64 __cdecl llrintf(float x); 
#line 3498 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.h"
extern double __cdecl nearbyint(double x); 
#line 3551 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.h"
extern float __cdecl nearbyintf(float x); 
#line 3611 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.h"
extern double __cdecl ceil(double x); 
#line 3663 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.h"
extern double __cdecl trunc(double x); 
#line 3716 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.h"
extern float __cdecl truncf(float x); 
#line 3742 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.h"
extern double __cdecl fdim(double x, double y); 
#line 3768 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.h"
extern float __cdecl fdimf(float x, float y); 
#line 4066 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.h"
extern double __cdecl atan2(double y, double x); 
#line 4137
extern double __cdecl atan(double x); 
#line 4160
extern double __cdecl acos(double x); 
#line 4211
extern double __cdecl asin(double x); 
#line 4276 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.h"
extern double __cdecl hypot(double x, double y); 
#line 4400 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.h"
static __inline float __cdecl hypotf(float x, float y); 
#line 5190 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.h"
extern double __cdecl cbrt(double x); 
#line 5276 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.h"
extern float __cdecl cbrtf(float x); 
#line 5329 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.h"
extern double rcbrt(double x); 
#line 5379
extern float rcbrtf(float x); 
#line 5439
extern double sinpi(double x); 
#line 5499
extern float sinpif(float x); 
#line 5551
extern double cospi(double x); 
#line 5603
extern float cospif(float x); 
#line 5633
extern void sincospi(double x, double * sptr, double * cptr); 
#line 5663
extern void sincospif(float x, float * sptr, float * cptr); 
#line 5996 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.h"
extern double __cdecl pow(double x, double y); 
#line 6052
extern double __cdecl modf(double x, double * iptr); 
#line 6111
extern double __cdecl fmod(double x, double y); 
#line 6209 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.h"
extern double __cdecl remainder(double x, double y); 
#line 6308 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.h"
extern float __cdecl remainderf(float x, float y); 
#line 6380 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.h"
extern double __cdecl remquo(double x, double y, int * quo); 
#line 6452 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.h"
extern float __cdecl remquof(float x, float y, int * quo); 
#line 6491 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.h"
extern double __cdecl j0(double x); 
#line 6533 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.h"
extern float j0f(float x); 
#line 6602 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.h"
extern double __cdecl j1(double x); 
#line 6671 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.h"
extern float j1f(float x); 
#line 6714 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.h"
extern double __cdecl jn(int n, double x); 
#line 6757 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.h"
extern float jnf(int n, float x); 
#line 6818 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.h"
extern double __cdecl y0(double x); 
#line 6879 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.h"
extern float y0f(float x); 
#line 6940 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.h"
extern double __cdecl y1(double x); 
#line 7001 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.h"
extern float y1f(float x); 
#line 7064 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.h"
extern double __cdecl yn(int n, double x); 
#line 7127 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.h"
extern float ynf(int n, float x); 
#line 7318 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.h"
extern double __cdecl erf(double x); 
#line 7400 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.h"
extern float __cdecl erff(float x); 
#line 7470 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.h"
extern double erfinv(double x); 
#line 7535
extern float erfinvf(float x); 
#line 7576 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.h"
extern double __cdecl erfc(double x); 
#line 7614 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.h"
extern float __cdecl erfcf(float x); 
#line 7731 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.h"
extern double __cdecl lgamma(double x); 
#line 7791 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.h"
extern double erfcinv(double x); 
#line 7846
extern float erfcinvf(float x); 
#line 7914
extern double normcdfinv(double x); 
#line 7982
extern float normcdfinvf(float x); 
#line 8025
extern double normcdf(double x); 
#line 8068
extern float normcdff(float x); 
#line 8132
extern double erfcx(double x); 
#line 8196
extern float erfcxf(float x); 
#line 8317 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.h"
extern float __cdecl lgammaf(float x); 
#line 8415 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.h"
extern double __cdecl tgamma(double x); 
#line 8513 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.h"
extern float __cdecl tgammaf(float x); 
#line 8526 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.h"
extern double __cdecl copysign(double x, double y); 
#line 8539 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.h"
extern float __cdecl copysignf(float x, float y); 
#line 8558 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.h"
extern double __cdecl nextafter(double x, double y); 
#line 8577 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.h"
extern float __cdecl nextafterf(float x, float y); 
#line 8593 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.h"
extern double __cdecl nan(const char * tagp); 
#line 8609 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.h"
extern float __cdecl nanf(const char * tagp); 
#line 8614 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.h"
extern int __isinff(float); 
#line 8615
extern int __isnanf(float); 
#line 8625 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.h"
extern int __finite(double); 
#line 8626
extern int __finitef(float); 
#line 8627
extern int __signbit(double); 
#line 8628
extern int __isnan(double); 
#line 8629
extern int __isinf(double); 
#line 8632 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.h"
extern int __signbitf(float); 
#line 8793 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.h"
extern double __cdecl fma(double x, double y, double z); 
#line 8951 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.h"
extern float __cdecl fmaf(float x, float y, float z); 
#line 8960 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.h"
extern int __signbitl(long double); 
#line 8966 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.h"
extern int __finitel(long double); 
#line 8967
extern int __isinfl(long double); 
#line 8968
extern int __isnanl(long double); 
#line 8972 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.h"
extern float __cdecl acosf(float); 
#line 8973
extern float __cdecl asinf(float); 
#line 8974
extern float __cdecl atanf(float); 
#line 8975
extern float __cdecl atan2f(float, float); 
#line 8976
extern float __cdecl cosf(float); 
#line 8977
extern float __cdecl sinf(float); 
#line 8978
extern float __cdecl tanf(float); 
#line 8979
extern float __cdecl coshf(float); 
#line 8980
extern float __cdecl sinhf(float); 
#line 8981
extern float __cdecl tanhf(float); 
#line 8982
extern float __cdecl expf(float); 
#line 8983
extern float __cdecl logf(float); 
#line 8984
extern float __cdecl log10f(float); 
#line 8985
extern float __cdecl modff(float, float *); 
#line 8986
extern float __cdecl powf(float, float); 
#line 8987
extern float __cdecl sqrtf(float); 
#line 8988
extern float __cdecl ceilf(float); 
#line 8989
extern float __cdecl floorf(float); 
#line 8990
extern float __cdecl fmodf(float, float); 
#line 10544 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.h"
}
#line 14 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\corecrt_math.h"
#pragma warning(push)
#pragma warning(disable: 4324 4514 4574 4710 4793 4820 4995 4996 28719 28726 28727 )
#line 18
__pragma( pack ( push, 8 )) extern "C" {
#line 23
struct _exception { 
#line 25
int type; 
#line 26
char *name; 
#line 27
double arg1; 
#line 28
double arg2; 
#line 29
double retval; 
#line 30
}; 
#line 37
struct _complex { 
#line 39
double x, y; 
#line 40
}; 
#line 59 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\corecrt_math.h"
typedef float float_t; 
#line 60
typedef double double_t; 
#line 78 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\corecrt_math.h"
extern const double _HUGE; 
#line 169 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\corecrt_math.h"
void __cdecl _fperrraise(int _Except); 
#line 171
short __cdecl _dclass(double _X); 
#line 172
short __cdecl _ldclass(long double _X); 
#line 173
short __cdecl _fdclass(float _X); 
#line 175
int __cdecl _dsign(double _X); 
#line 176
int __cdecl _ldsign(long double _X); 
#line 177
int __cdecl _fdsign(float _X); 
#line 179
int __cdecl _dpcomp(double _X, double _Y); 
#line 180
int __cdecl _ldpcomp(long double _X, long double _Y); 
#line 181
int __cdecl _fdpcomp(float _X, float _Y); 
#line 183
short __cdecl _dtest(double * _Px); 
#line 184
short __cdecl _ldtest(long double * _Px); 
#line 185
short __cdecl _fdtest(float * _Px); 
#line 187
short __cdecl _d_int(double * _Px, short _Xexp); 
#line 188
short __cdecl _ld_int(long double * _Px, short _Xexp); 
#line 189
short __cdecl _fd_int(float * _Px, short _Xexp); 
#line 191
short __cdecl _dscale(double * _Px, long _Lexp); 
#line 192
short __cdecl _ldscale(long double * _Px, long _Lexp); 
#line 193
short __cdecl _fdscale(float * _Px, long _Lexp); 
#line 195
short __cdecl _dunscale(short * _Pex, double * _Px); 
#line 196
short __cdecl _ldunscale(short * _Pex, long double * _Px); 
#line 197
short __cdecl _fdunscale(short * _Pex, float * _Px); 
#line 199
short __cdecl _dexp(double * _Px, double _Y, long _Eoff); 
#line 200
short __cdecl _ldexp(long double * _Px, long double _Y, long _Eoff); 
#line 201
short __cdecl _fdexp(float * _Px, float _Y, long _Eoff); 
#line 203
short __cdecl _dnorm(unsigned short * _Ps); 
#line 204
short __cdecl _fdnorm(unsigned short * _Ps); 
#line 206
double __cdecl _dpoly(double _X, const double * _Tab, int _N); 
#line 207
long double __cdecl _ldpoly(long double _X, const long double * _Tab, int _N); 
#line 208
float __cdecl _fdpoly(float _X, const float * _Tab, int _N); 
#line 210
double __cdecl _dlog(double _X, int _Baseflag); 
#line 211
long double __cdecl _ldlog(long double _X, int _Baseflag); 
#line 212
float __cdecl _fdlog(float _X, int _Baseflag); 
#line 214
double __cdecl _dsin(double _X, unsigned _Qoff); 
#line 215
long double __cdecl _ldsin(long double _X, unsigned _Qoff); 
#line 216
float __cdecl _fdsin(float _X, unsigned _Qoff); 
#line 223
typedef 
#line 220
union { 
#line 221
unsigned short _Sh[4]; 
#line 222
double _Val; 
#line 223
} _double_val; 
#line 230
typedef 
#line 227
union { 
#line 228
unsigned short _Sh[2]; 
#line 229
float _Val; 
#line 230
} _float_val; 
#line 237
typedef 
#line 234
union { 
#line 235
unsigned short _Sh[4]; 
#line 236
long double _Val; 
#line 237
} _ldouble_val; 
#line 245
typedef 
#line 240
union { 
#line 241
unsigned short _Word[4]; 
#line 242
float _Float; 
#line 243
double _Double; 
#line 244
long double _Long_double; 
#line 245
} _float_const; 
#line 247
extern const _float_const _Denorm_C, _Inf_C, _Nan_C, _Snan_C, _Hugeval_C; 
#line 248
extern const _float_const _FDenorm_C, _FInf_C, _FNan_C, _FSnan_C; 
#line 249
extern const _float_const _LDenorm_C, _LInf_C, _LNan_C, _LSnan_C; 
#line 251
extern const _float_const _Eps_C, _Rteps_C; 
#line 252
extern const _float_const _FEps_C, _FRteps_C; 
#line 253
extern const _float_const _LEps_C, _LRteps_C; 
#line 255
extern const double _Zero_C, _Xbig_C; 
#line 256
extern const float _FZero_C, _FXbig_C; 
#line 257
extern const long double _LZero_C, _LXbig_C; 
#line 286
extern "C++" {
#line 288
inline int fpclassify(float _X) throw() 
#line 289
{ 
#line 290
return _fdtest(&_X); 
#line 291
} 
#line 293
inline int fpclassify(double _X) throw() 
#line 294
{ 
#line 295
return _dtest(&_X); 
#line 296
} 
#line 298
inline int fpclassify(long double _X) throw() 
#line 299
{ 
#line 300
return _ldtest(&_X); 
#line 301
} 
#line 303
inline bool signbit(float _X) throw() 
#line 304
{ 
#line 305
return _fdsign(_X) != 0; 
#line 306
} 
#line 308
inline bool signbit(double _X) throw() 
#line 309
{ 
#line 310
return _dsign(_X) != 0; 
#line 311
} 
#line 313
inline bool signbit(long double _X) throw() 
#line 314
{ 
#line 315
return _ldsign(_X) != 0; 
#line 316
} 
#line 318
inline int _fpcomp(float _X, float _Y) throw() 
#line 319
{ 
#line 320
return _fdpcomp(_X, _Y); 
#line 321
} 
#line 323
inline int _fpcomp(double _X, double _Y) throw() 
#line 324
{ 
#line 325
return _dpcomp(_X, _Y); 
#line 326
} 
#line 328
inline int _fpcomp(long double _X, long double _Y) throw() 
#line 329
{ 
#line 330
return _ldpcomp(_X, _Y); 
#line 331
} 
#line 333
template< class _Trc, class _Tre> struct _Combined_type { 
#line 335
typedef float _Type; 
#line 336
}; 
#line 338
template<> struct _Combined_type< float, double>  { 
#line 340
typedef double _Type; 
#line 341
}; 
#line 343
template<> struct _Combined_type< float, long double>  { 
#line 345
typedef long double _Type; 
#line 346
}; 
#line 348
template< class _Ty, class _T2> struct _Real_widened { 
#line 350
typedef long double _Type; 
#line 351
}; 
#line 353
template<> struct _Real_widened< float, float>  { 
#line 355
typedef float _Type; 
#line 356
}; 
#line 358
template<> struct _Real_widened< float, double>  { 
#line 360
typedef double _Type; 
#line 361
}; 
#line 363
template<> struct _Real_widened< double, float>  { 
#line 365
typedef double _Type; 
#line 366
}; 
#line 368
template<> struct _Real_widened< double, double>  { 
#line 370
typedef double _Type; 
#line 371
}; 
#line 373
template< class _Ty> struct _Real_type { 
#line 375
typedef double _Type; 
#line 376
}; 
#line 378
template<> struct _Real_type< float>  { 
#line 380
typedef float _Type; 
#line 381
}; 
#line 383
template<> struct _Real_type< long double>  { 
#line 385
typedef long double _Type; 
#line 386
}; 
#line 388
template < class _T1, class _T2 >
      inline int _fpcomp ( _T1 _X, _T2 _Y ) throw ( )
    {
        typedef typename _Combined_type < float,
            typename _Real_widened <
            typename _Real_type < _T1 > :: _Type,
            typename _Real_type < _T2 > :: _Type > :: _Type > :: _Type _Tw;
        return _fpcomp ( ( _Tw ) _X, ( _Tw ) _Y );
    }
#line 398
template < class _Ty >
      inline bool isfinite ( _Ty _X ) throw ( )
    {
        return fpclassify ( _X ) <= 0;
    }
#line 404
template < class _Ty >
      inline bool isinf ( _Ty _X ) throw ( )
    {
        return fpclassify ( _X ) == 1;
    }
#line 410
template < class _Ty >
      inline bool isnan ( _Ty _X ) throw ( )
    {
        return fpclassify ( _X ) == 2;
    }
#line 416
template < class _Ty >
      inline bool isnormal ( _Ty _X ) throw ( )
    {
        return fpclassify ( _X ) == ( - 1 );
    }
#line 422
template < class _Ty1, class _Ty2 >
      inline bool isgreater ( _Ty1 _X, _Ty2 _Y ) throw ( )
    {
        return ( _fpcomp ( _X, _Y ) & 4 ) != 0;
    }
#line 428
template < class _Ty1, class _Ty2 >
      inline bool isgreaterequal ( _Ty1 _X, _Ty2 _Y ) throw ( )
    {
        return ( _fpcomp ( _X, _Y ) & ( 2 | 4 ) ) != 0;
    }
#line 434
template < class _Ty1, class _Ty2 >
      inline bool isless ( _Ty1 _X, _Ty2 _Y ) throw ( )
    {
        return ( _fpcomp ( _X, _Y ) & 1 ) != 0;
    }
#line 440
template < class _Ty1, class _Ty2 >
      inline bool islessequal ( _Ty1 _X, _Ty2 _Y ) throw ( )
    {
        return ( _fpcomp ( _X, _Y ) & ( 1 | 2 ) ) != 0;
    }
#line 446
template < class _Ty1, class _Ty2 >
      inline bool islessgreater ( _Ty1 _X, _Ty2 _Y ) throw ( )
    {
        return ( _fpcomp ( _X, _Y ) & ( 1 | 4 ) ) != 0;
    }
#line 452
template < class _Ty1, class _Ty2 >
      inline bool isunordered ( _Ty1 _X, _Ty2 _Y ) throw ( )
    {
        return _fpcomp ( _X, _Y ) == 0;
    }
#line 457
}
#line 464 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\corecrt_math.h"
int __cdecl abs(int _X); 
#line 465
long __cdecl labs(long _X); 
#line 466
__int64 __cdecl llabs(__int64 _X); 
#line 468
double __cdecl acos(double _X); 
#line 469
double __cdecl asin(double _X); 
#line 470
double __cdecl atan(double _X); 
#line 471
double __cdecl atan2(double _Y, double _X); 
#line 473
double __cdecl cos(double _X); 
#line 474
double __cdecl cosh(double _X); 
#line 475
double __cdecl exp(double _X); 
#line 476
double __cdecl fabs(double _X); 
#line 477
double __cdecl fmod(double _X, double _Y); 
#line 478
double __cdecl log(double _X); 
#line 479
double __cdecl log10(double _X); 
#line 480
double __cdecl pow(double _X, double _Y); 
#line 481
double __cdecl sin(double _X); 
#line 482
double __cdecl sinh(double _X); 
#line 483
double __cdecl sqrt(double _X); 
#line 484
double __cdecl tan(double _X); 
#line 485
double __cdecl tanh(double _X); 
#line 487
double __cdecl acosh(double _X); 
#line 488
double __cdecl asinh(double _X); 
#line 489
double __cdecl atanh(double _X); 
#line 490
double __cdecl atof(const char * _String); 
#line 491
double __cdecl _atof_l(const char * _String, _locale_t _Locale); 
#line 492
double __cdecl _cabs(_complex _Complex_value); 
#line 493
double __cdecl cbrt(double _X); 
#line 494
double __cdecl ceil(double _X); 
#line 495
double __cdecl _chgsign(double _X); 
#line 496
double __cdecl copysign(double _Number, double _Sign); 
#line 497
double __cdecl _copysign(double _Number, double _Sign); 
#line 498
double __cdecl erf(double _X); 
#line 499
double __cdecl erfc(double _X); 
#line 500
double __cdecl exp2(double _X); 
#line 501
double __cdecl expm1(double _X); 
#line 502
double __cdecl fdim(double _X, double _Y); 
#line 503
double __cdecl floor(double _X); 
#line 504
double __cdecl fma(double _X, double _Y, double _Z); 
#line 505
double __cdecl fmax(double _X, double _Y); 
#line 506
double __cdecl fmin(double _X, double _Y); 
#line 507
double __cdecl frexp(double _X, int * _Y); 
#line 508
double __cdecl hypot(double _X, double _Y); 
#line 509
double __cdecl _hypot(double _X, double _Y); 
#line 510
int __cdecl ilogb(double _X); 
#line 511
double __cdecl ldexp(double _X, int _Y); 
#line 512
double __cdecl lgamma(double _X); 
#line 513
__int64 __cdecl llrint(double _X); 
#line 514
__int64 __cdecl llround(double _X); 
#line 515
double __cdecl log1p(double _X); 
#line 516
double __cdecl log2(double _X); 
#line 517
double __cdecl logb(double _X); 
#line 518
long __cdecl lrint(double _X); 
#line 519
long __cdecl lround(double _X); 
#line 521
int __cdecl _matherr(_exception * _Except); 
#line 523
double __cdecl modf(double _X, double * _Y); 
#line 524
double __cdecl nan(const char * _X); 
#line 525
double __cdecl nearbyint(double _X); 
#line 526
double __cdecl nextafter(double _X, double _Y); 
#line 527
double __cdecl nexttoward(double _X, long double _Y); 
#line 528
double __cdecl remainder(double _X, double _Y); 
#line 529
double __cdecl remquo(double _X, double _Y, int * _Z); 
#line 530
double __cdecl rint(double _X); 
#line 531
double __cdecl round(double _X); 
#line 532
double __cdecl scalbln(double _X, long _Y); 
#line 533
double __cdecl scalbn(double _X, int _Y); 
#line 534
double __cdecl tgamma(double _X); 
#line 535
double __cdecl trunc(double _X); 
#line 536
double __cdecl _j0(double _X); 
#line 537
double __cdecl _j1(double _X); 
#line 538
double __cdecl _jn(int _X, double _Y); 
#line 539
double __cdecl _y0(double _X); 
#line 540
double __cdecl _y1(double _X); 
#line 541
double __cdecl _yn(int _X, double _Y); 
#line 543
float __cdecl acoshf(float _X); 
#line 544
float __cdecl asinhf(float _X); 
#line 545
float __cdecl atanhf(float _X); 
#line 546
float __cdecl cbrtf(float _X); 
#line 547
float __cdecl _chgsignf(float _X); 
#line 548
float __cdecl copysignf(float _Number, float _Sign); 
#line 549
float __cdecl _copysignf(float _Number, float _Sign); 
#line 550
float __cdecl erff(float _X); 
#line 551
float __cdecl erfcf(float _X); 
#line 552
float __cdecl expm1f(float _X); 
#line 553
float __cdecl exp2f(float _X); 
#line 554
float __cdecl fdimf(float _X, float _Y); 
#line 555
float __cdecl fmaf(float _X, float _Y, float _Z); 
#line 556
float __cdecl fmaxf(float _X, float _Y); 
#line 557
float __cdecl fminf(float _X, float _Y); 
#line 558
float __cdecl _hypotf(float _X, float _Y); 
#line 559
int __cdecl ilogbf(float _X); 
#line 560
float __cdecl lgammaf(float _X); 
#line 561
__int64 __cdecl llrintf(float _X); 
#line 562
__int64 __cdecl llroundf(float _X); 
#line 563
float __cdecl log1pf(float _X); 
#line 564
float __cdecl log2f(float _X); 
#line 565
float __cdecl logbf(float _X); 
#line 566
long __cdecl lrintf(float _X); 
#line 567
long __cdecl lroundf(float _X); 
#line 568
float __cdecl nanf(const char * _X); 
#line 569
float __cdecl nearbyintf(float _X); 
#line 570
float __cdecl nextafterf(float _X, float _Y); 
#line 571
float __cdecl nexttowardf(float _X, long double _Y); 
#line 572
float __cdecl remainderf(float _X, float _Y); 
#line 573
float __cdecl remquof(float _X, float _Y, int * _Z); 
#line 574
float __cdecl rintf(float _X); 
#line 575
float __cdecl roundf(float _X); 
#line 576
float __cdecl scalblnf(float _X, long _Y); 
#line 577
float __cdecl scalbnf(float _X, int _Y); 
#line 578
float __cdecl tgammaf(float _X); 
#line 579
float __cdecl truncf(float _X); 
#line 589 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\corecrt_math.h"
float __cdecl _logbf(float _X); 
#line 590
float __cdecl _nextafterf(float _X, float _Y); 
#line 591
int __cdecl _finitef(float _X); 
#line 592
int __cdecl _isnanf(float _X); 
#line 593
int __cdecl _fpclassf(float _X); 
#line 595
int __cdecl _set_FMA3_enable(int _Flag); 
#line 596
int __cdecl _get_FMA3_enable(); 
#line 609 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\corecrt_math.h"
float __cdecl acosf(float _X); 
#line 610
float __cdecl asinf(float _X); 
#line 611
float __cdecl atan2f(float _Y, float _X); 
#line 612
float __cdecl atanf(float _X); 
#line 613
float __cdecl ceilf(float _X); 
#line 614
float __cdecl cosf(float _X); 
#line 615
float __cdecl coshf(float _X); 
#line 616
float __cdecl expf(float _X); 
#line 668 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\corecrt_math.h"
__inline float __cdecl fabsf(float _X) 
#line 669
{ 
#line 670
return (float)fabs(_X); 
#line 671
} 
#line 677 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\corecrt_math.h"
float __cdecl floorf(float _X); 
#line 678
float __cdecl fmodf(float _X, float _Y); 
#line 694 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\corecrt_math.h"
__inline float __cdecl frexpf(float _X, int *_Y) 
#line 695
{ 
#line 696
return (float)frexp(_X, _Y); 
#line 697
} 
#line 699
__inline float __cdecl hypotf(float _X, float _Y) 
#line 700
{ 
#line 701
return _hypotf(_X, _Y); 
#line 702
} 
#line 704
__inline float __cdecl ldexpf(float _X, int _Y) 
#line 705
{ 
#line 706
return (float)ldexp(_X, _Y); 
#line 707
} 
#line 711
float __cdecl log10f(float _X); 
#line 712
float __cdecl logf(float _X); 
#line 713
float __cdecl modff(float _X, float * _Y); 
#line 714
float __cdecl powf(float _X, float _Y); 
#line 715
float __cdecl sinf(float _X); 
#line 716
float __cdecl sinhf(float _X); 
#line 717
float __cdecl sqrtf(float _X); 
#line 718
float __cdecl tanf(float _X); 
#line 719
float __cdecl tanhf(float _X); 
#line 773 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\corecrt_math.h"
long double __cdecl acoshl(long double _X); 
#line 775
__inline long double __cdecl acosl(long double _X) 
#line 776
{ 
#line 777
return acos((double)_X); 
#line 778
} 
#line 780
long double __cdecl asinhl(long double _X); 
#line 782
__inline long double __cdecl asinl(long double _X) 
#line 783
{ 
#line 784
return asin((double)_X); 
#line 785
} 
#line 787
__inline long double __cdecl atan2l(long double _Y, long double _X) 
#line 788
{ 
#line 789
return atan2((double)_Y, (double)_X); 
#line 790
} 
#line 792
long double __cdecl atanhl(long double _X); 
#line 794
__inline long double __cdecl atanl(long double _X) 
#line 795
{ 
#line 796
return atan((double)_X); 
#line 797
} 
#line 799
long double __cdecl cbrtl(long double _X); 
#line 801
__inline long double __cdecl ceill(long double _X) 
#line 802
{ 
#line 803
return ceil((double)_X); 
#line 804
} 
#line 806
__inline long double __cdecl _chgsignl(long double _X) 
#line 807
{ 
#line 808
return _chgsign((double)_X); 
#line 809
} 
#line 811
long double __cdecl copysignl(long double _Number, long double _Sign); 
#line 813
__inline long double __cdecl _copysignl(long double _Number, long double _Sign) 
#line 814
{ 
#line 815
return _copysign((double)_Number, (double)_Sign); 
#line 816
} 
#line 818
__inline long double __cdecl coshl(long double _X) 
#line 819
{ 
#line 820
return cosh((double)_X); 
#line 821
} 
#line 823
__inline long double __cdecl cosl(long double _X) 
#line 824
{ 
#line 825
return cos((double)_X); 
#line 826
} 
#line 828
long double __cdecl erfl(long double _X); 
#line 829
long double __cdecl erfcl(long double _X); 
#line 831
__inline long double __cdecl expl(long double _X) 
#line 832
{ 
#line 833
return exp((double)_X); 
#line 834
} 
#line 836
long double __cdecl exp2l(long double _X); 
#line 837
long double __cdecl expm1l(long double _X); 
#line 839
__inline long double __cdecl fabsl(long double _X) 
#line 840
{ 
#line 841
return fabs((double)_X); 
#line 842
} 
#line 844
long double __cdecl fdiml(long double _X, long double _Y); 
#line 846
__inline long double __cdecl floorl(long double _X) 
#line 847
{ 
#line 848
return floor((double)_X); 
#line 849
} 
#line 851
long double __cdecl fmal(long double _X, long double _Y, long double _Z); 
#line 852
long double __cdecl fmaxl(long double _X, long double _Y); 
#line 853
long double __cdecl fminl(long double _X, long double _Y); 
#line 855
__inline long double __cdecl fmodl(long double _X, long double _Y) 
#line 856
{ 
#line 857
return fmod((double)_X, (double)_Y); 
#line 858
} 
#line 860
__inline long double __cdecl frexpl(long double _X, int *_Y) 
#line 861
{ 
#line 862
return frexp((double)_X, _Y); 
#line 863
} 
#line 865
int __cdecl ilogbl(long double _X); 
#line 867
__inline long double __cdecl _hypotl(long double _X, long double _Y) 
#line 868
{ 
#line 869
return _hypot((double)_X, (double)_Y); 
#line 870
} 
#line 872
__inline long double __cdecl hypotl(long double _X, long double _Y) 
#line 873
{ 
#line 874
return _hypot((double)_X, (double)_Y); 
#line 875
} 
#line 877
__inline long double __cdecl ldexpl(long double _X, int _Y) 
#line 878
{ 
#line 879
return ldexp((double)_X, _Y); 
#line 880
} 
#line 882
long double __cdecl lgammal(long double _X); 
#line 883
__int64 __cdecl llrintl(long double _X); 
#line 884
__int64 __cdecl llroundl(long double _X); 
#line 886
__inline long double __cdecl logl(long double _X) 
#line 887
{ 
#line 888
return log((double)_X); 
#line 889
} 
#line 891
__inline long double __cdecl log10l(long double _X) 
#line 892
{ 
#line 893
return log10((double)_X); 
#line 894
} 
#line 896
long double __cdecl log1pl(long double _X); 
#line 897
long double __cdecl log2l(long double _X); 
#line 898
long double __cdecl logbl(long double _X); 
#line 899
long __cdecl lrintl(long double _X); 
#line 900
long __cdecl lroundl(long double _X); 
#line 902
__inline long double __cdecl modfl(long double _X, long double *_Y) 
#line 903
{ 
#line 904
double _F, _I; 
#line 905
_F = modf((double)_X, &_I); 
#line 906
(*_Y) = _I; 
#line 907
return _F; 
#line 908
} 
#line 910
long double __cdecl nanl(const char * _X); 
#line 911
long double __cdecl nearbyintl(long double _X); 
#line 912
long double __cdecl nextafterl(long double _X, long double _Y); 
#line 913
long double __cdecl nexttowardl(long double _X, long double _Y); 
#line 915
__inline long double __cdecl powl(long double _X, long double _Y) 
#line 916
{ 
#line 917
return pow((double)_X, (double)_Y); 
#line 918
} 
#line 920
long double __cdecl remainderl(long double _X, long double _Y); 
#line 921
long double __cdecl remquol(long double _X, long double _Y, int * _Z); 
#line 922
long double __cdecl rintl(long double _X); 
#line 923
long double __cdecl roundl(long double _X); 
#line 924
long double __cdecl scalblnl(long double _X, long _Y); 
#line 925
long double __cdecl scalbnl(long double _X, int _Y); 
#line 927
__inline long double __cdecl sinhl(long double _X) 
#line 928
{ 
#line 929
return sinh((double)_X); 
#line 930
} 
#line 932
__inline long double __cdecl sinl(long double _X) 
#line 933
{ 
#line 934
return sin((double)_X); 
#line 935
} 
#line 937
__inline long double __cdecl sqrtl(long double _X) 
#line 938
{ 
#line 939
return sqrt((double)_X); 
#line 940
} 
#line 942
__inline long double __cdecl tanhl(long double _X) 
#line 943
{ 
#line 944
return tanh((double)_X); 
#line 945
} 
#line 947
__inline long double __cdecl tanl(long double _X) 
#line 948
{ 
#line 949
return tan((double)_X); 
#line 950
} 
#line 952
long double __cdecl tgammal(long double _X); 
#line 953
long double __cdecl truncl(long double _X); 
#line 974 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\corecrt_math.h"
extern double HUGE; 
#line 979 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\corecrt_math.h"
double __cdecl j0(double _X); 
#line 980
double __cdecl j1(double _X); 
#line 981
double __cdecl jn(int _X, double _Y); 
#line 982
double __cdecl y0(double _X); 
#line 983
double __cdecl y1(double _X); 
#line 984
double __cdecl yn(int _X, double _Y); 
#line 989 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\corecrt_math.h"
}__pragma( pack ( pop )) 
#line 991
#pragma warning(pop)
#line 13 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\corecrt_malloc.h"
#pragma warning(push)
#pragma warning(disable: 4324 4514 4574 4710 4793 4820 4995 4996 28719 28726 28727 )
#line 17
__pragma( pack ( push, 8 )) extern "C" {
#line 58 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\corecrt_malloc.h"
__declspec(allocator) __declspec(restrict) void *__cdecl 
#line 59
_calloc_base(size_t _Count, size_t _Size); 
#line 65
__declspec(allocator) __declspec(restrict) void *__cdecl 
#line 66
calloc(size_t _Count, size_t _Size); 
#line 72
int __cdecl _callnewh(size_t _Size); 
#line 77
__declspec(allocator) void *__cdecl 
#line 78
_expand(void * _Block, size_t _Size); 
#line 84
void __cdecl _free_base(void * _Block); 
#line 89
void __cdecl free(void * _Block); 
#line 94
__declspec(allocator) __declspec(restrict) void *__cdecl 
#line 95
_malloc_base(size_t _Size); 
#line 100
__declspec(allocator) __declspec(restrict) void *__cdecl 
#line 101
malloc(size_t _Size); 
#line 107
size_t __cdecl _msize_base(void * _Block); 
#line 113
size_t __cdecl _msize(void * _Block); 
#line 118
__declspec(allocator) __declspec(restrict) void *__cdecl 
#line 119
_realloc_base(void * _Block, size_t _Size); 
#line 125
__declspec(allocator) __declspec(restrict) void *__cdecl 
#line 126
realloc(void * _Block, size_t _Size); 
#line 132
__declspec(allocator) __declspec(restrict) void *__cdecl 
#line 133
_recalloc_base(void * _Block, size_t _Count, size_t _Size); 
#line 140
__declspec(allocator) __declspec(restrict) void *__cdecl 
#line 141
_recalloc(void * _Block, size_t _Count, size_t _Size); 
#line 148
void __cdecl _aligned_free(void * _Block); 
#line 153
__declspec(allocator) __declspec(restrict) void *__cdecl 
#line 154
_aligned_malloc(size_t _Size, size_t _Alignment); 
#line 160
__declspec(allocator) __declspec(restrict) void *__cdecl 
#line 161
_aligned_offset_malloc(size_t _Size, size_t _Alignment, size_t _Offset); 
#line 169
size_t __cdecl _aligned_msize(void * _Block, size_t _Alignment, size_t _Offset); 
#line 176
__declspec(allocator) __declspec(restrict) void *__cdecl 
#line 177
_aligned_offset_realloc(void * _Block, size_t _Size, size_t _Alignment, size_t _Offset); 
#line 185
__declspec(allocator) __declspec(restrict) void *__cdecl 
#line 186
_aligned_offset_recalloc(void * _Block, size_t _Count, size_t _Size, size_t _Alignment, size_t _Offset); 
#line 195
__declspec(allocator) __declspec(restrict) void *__cdecl 
#line 196
_aligned_realloc(void * _Block, size_t _Size, size_t _Alignment); 
#line 203
__declspec(allocator) __declspec(restrict) void *__cdecl 
#line 204
_aligned_recalloc(void * _Block, size_t _Count, size_t _Size, size_t _Alignment); 
#line 232 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\corecrt_malloc.h"
}__pragma( pack ( pop )) 
#line 234
#pragma warning(pop)
#line 16 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\corecrt_search.h"
#pragma warning(push)
#pragma warning(disable: 4324 4514 4574 4710 4793 4820 4995 4996 28719 28726 28727 )
#line 20
__pragma( pack ( push, 8 )) extern "C" {
#line 23
typedef int (__cdecl *_CoreCrtSecureSearchSortCompareFunction)(void *, const void *, const void *); 
#line 24
typedef int (__cdecl *_CoreCrtNonSecureSearchSortCompareFunction)(const void *, const void *); 
#line 30
void *__cdecl bsearch_s(const void * _Key, const void * _Base, rsize_t _NumOfElements, rsize_t _SizeOfElements, _CoreCrtSecureSearchSortCompareFunction _CompareFunction, void * _Context); 
#line 39
void __cdecl qsort_s(void * _Base, rsize_t _NumOfElements, rsize_t _SizeOfElements, _CoreCrtSecureSearchSortCompareFunction _CompareFunction, void * _Context); 
#line 52 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\corecrt_search.h"
void *__cdecl bsearch(const void * _Key, const void * _Base, size_t _NumOfElements, size_t _SizeOfElements, _CoreCrtNonSecureSearchSortCompareFunction _CompareFunction); 
#line 60
void __cdecl qsort(void * _Base, size_t _NumOfElements, size_t _SizeOfElements, _CoreCrtNonSecureSearchSortCompareFunction _CompareFunction); 
#line 68
void *__cdecl _lfind_s(const void * _Key, const void * _Base, unsigned * _NumOfElements, size_t _SizeOfElements, _CoreCrtSecureSearchSortCompareFunction _CompareFunction, void * _Context); 
#line 78
void *__cdecl _lfind(const void * _Key, const void * _Base, unsigned * _NumOfElements, unsigned _SizeOfElements, _CoreCrtNonSecureSearchSortCompareFunction _CompareFunction); 
#line 87
void *__cdecl _lsearch_s(const void * _Key, void * _Base, unsigned * _NumOfElements, size_t _SizeOfElements, _CoreCrtSecureSearchSortCompareFunction _CompareFunction, void * _Context); 
#line 97
void *__cdecl _lsearch(const void * _Key, void * _Base, unsigned * _NumOfElements, unsigned _SizeOfElements, _CoreCrtNonSecureSearchSortCompareFunction _CompareFunction); 
#line 195 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\corecrt_search.h"
void *__cdecl lfind(const void * _Key, const void * _Base, unsigned * _NumOfElements, unsigned _SizeOfElements, _CoreCrtNonSecureSearchSortCompareFunction _CompareFunction); 
#line 204
void *__cdecl lsearch(const void * _Key, void * _Base, unsigned * _NumOfElements, unsigned _SizeOfElements, _CoreCrtNonSecureSearchSortCompareFunction _CompareFunction); 
#line 216 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\corecrt_search.h"
}__pragma( pack ( pop )) 
#line 218
#pragma warning(pop)
#line 13 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\corecrt_wstdlib.h"
#pragma warning(push)
#pragma warning(disable: 4324 4514 4574 4710 4793 4820 4995 4996 28719 28726 28727 )
#line 17
__pragma( pack ( push, 8 )) extern "C" {
#line 54
errno_t __cdecl _itow_s(int _Value, __wchar_t * _Buffer, size_t _BufferCount, int _Radix); 
#line 61
extern "C++" {template < size_t _Size > inline errno_t __cdecl _itow_s ( int _Value, wchar_t ( & _Buffer ) [ _Size ], int _Radix ) throw ( ) { return _itow_s ( _Value, _Buffer, _Size, _Radix ); }}
#line 68 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\corecrt_wstdlib.h"
__wchar_t *__cdecl _itow(int _Value, __wchar_t * _Buffer, int _Radix); 
#line 77 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\corecrt_wstdlib.h"
errno_t __cdecl _ltow_s(long _Value, __wchar_t * _Buffer, size_t _BufferCount, int _Radix); 
#line 84
extern "C++" {template < size_t _Size > inline errno_t __cdecl _ltow_s ( long _Value, wchar_t ( & _Buffer ) [ _Size ], int _Radix ) throw ( ) { return _ltow_s ( _Value, _Buffer, _Size, _Radix ); }}
#line 91 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\corecrt_wstdlib.h"
__wchar_t *__cdecl _ltow(long _Value, __wchar_t * _Buffer, int _Radix); 
#line 99 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\corecrt_wstdlib.h"
errno_t __cdecl _ultow_s(unsigned long _Value, __wchar_t * _Buffer, size_t _BufferCount, int _Radix); 
#line 106
extern "C++" {template < size_t _Size > inline errno_t __cdecl _ultow_s ( unsigned long _Value, wchar_t ( & _Buffer ) [ _Size ], int _Radix ) throw ( ) { return _ultow_s ( _Value, _Buffer, _Size, _Radix ); }}
#line 113 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\corecrt_wstdlib.h"
__wchar_t *__cdecl _ultow(unsigned long _Value, __wchar_t * _Buffer, int _Radix); 
#line 121 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\corecrt_wstdlib.h"
double __cdecl wcstod(const __wchar_t * _String, __wchar_t ** _EndPtr); 
#line 127
double __cdecl _wcstod_l(const __wchar_t * _String, __wchar_t ** _EndPtr, _locale_t _Locale); 
#line 134
long __cdecl wcstol(const __wchar_t * _String, __wchar_t ** _EndPtr, int _Radix); 
#line 141
long __cdecl _wcstol_l(const __wchar_t * _String, __wchar_t ** _EndPtr, int _Radix, _locale_t _Locale); 
#line 149
__int64 __cdecl wcstoll(const __wchar_t * _String, __wchar_t ** _EndPtr, int _Radix); 
#line 156
__int64 __cdecl _wcstoll_l(const __wchar_t * _String, __wchar_t ** _EndPtr, int _Radix, _locale_t _Locale); 
#line 164
unsigned long __cdecl wcstoul(const __wchar_t * _String, __wchar_t ** _EndPtr, int _Radix); 
#line 171
unsigned long __cdecl _wcstoul_l(const __wchar_t * _String, __wchar_t ** _EndPtr, int _Radix, _locale_t _Locale); 
#line 179
unsigned __int64 __cdecl wcstoull(const __wchar_t * _String, __wchar_t ** _EndPtr, int _Radix); 
#line 186
unsigned __int64 __cdecl _wcstoull_l(const __wchar_t * _String, __wchar_t ** _EndPtr, int _Radix, _locale_t _Locale); 
#line 194
long double __cdecl wcstold(const __wchar_t * _String, __wchar_t ** _EndPtr); 
#line 200
long double __cdecl _wcstold_l(const __wchar_t * _String, __wchar_t ** _EndPtr, _locale_t _Locale); 
#line 207
float __cdecl wcstof(const __wchar_t * _String, __wchar_t ** _EndPtr); 
#line 213
float __cdecl _wcstof_l(const __wchar_t * _String, __wchar_t ** _EndPtr, _locale_t _Locale); 
#line 220
double __cdecl _wtof(const __wchar_t * _String); 
#line 225
double __cdecl _wtof_l(const __wchar_t * _String, _locale_t _Locale); 
#line 231
int __cdecl _wtoi(const __wchar_t * _String); 
#line 236
int __cdecl _wtoi_l(const __wchar_t * _String, _locale_t _Locale); 
#line 242
long __cdecl _wtol(const __wchar_t * _String); 
#line 247
long __cdecl _wtol_l(const __wchar_t * _String, _locale_t _Locale); 
#line 253
__int64 __cdecl _wtoll(const __wchar_t * _String); 
#line 258
__int64 __cdecl _wtoll_l(const __wchar_t * _String, _locale_t _Locale); 
#line 264
errno_t __cdecl _i64tow_s(__int64 _Value, __wchar_t * _Buffer, size_t _BufferCount, int _Radix); 
#line 272
__wchar_t *__cdecl _i64tow(__int64 _Value, __wchar_t * _Buffer, int _Radix); 
#line 279
errno_t __cdecl _ui64tow_s(unsigned __int64 _Value, __wchar_t * _Buffer, size_t _BufferCount, int _Radix); 
#line 287
__wchar_t *__cdecl _ui64tow(unsigned __int64 _Value, __wchar_t * _Buffer, int _Radix); 
#line 294
__int64 __cdecl _wtoi64(const __wchar_t * _String); 
#line 299
__int64 __cdecl _wtoi64_l(const __wchar_t * _String, _locale_t _Locale); 
#line 305
__int64 __cdecl _wcstoi64(const __wchar_t * _String, __wchar_t ** _EndPtr, int _Radix); 
#line 312
__int64 __cdecl _wcstoi64_l(const __wchar_t * _String, __wchar_t ** _EndPtr, int _Radix, _locale_t _Locale); 
#line 320
unsigned __int64 __cdecl _wcstoui64(const __wchar_t * _String, __wchar_t ** _EndPtr, int _Radix); 
#line 327
unsigned __int64 __cdecl _wcstoui64_l(const __wchar_t * _String, __wchar_t ** _EndPtr, int _Radix, _locale_t _Locale); 
#line 339
__declspec(allocator) __wchar_t *__cdecl _wfullpath(__wchar_t * _Buffer, const __wchar_t * _Path, size_t _BufferCount); 
#line 348
errno_t __cdecl _wmakepath_s(__wchar_t * _Buffer, size_t _BufferCount, const __wchar_t * _Drive, const __wchar_t * _Dir, const __wchar_t * _Filename, const __wchar_t * _Ext); 
#line 357
extern "C++" {template < size_t _Size > inline errno_t __cdecl _wmakepath_s ( wchar_t ( & _Buffer ) [ _Size ], wchar_t const * _Drive, wchar_t const * _Dir, wchar_t const * _Filename, wchar_t const * _Ext ) throw ( ) { return _wmakepath_s ( _Buffer, _Size, _Drive, _Dir, _Filename, _Ext ); }}
#line 366 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\corecrt_wstdlib.h"
void __cdecl _wmakepath(__wchar_t * _Buffer, const __wchar_t * _Drive, const __wchar_t * _Dir, const __wchar_t * _Filename, const __wchar_t * _Ext); 
#line 375 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\corecrt_wstdlib.h"
void __cdecl _wperror(const __wchar_t * _ErrorMessage); 
#line 380
void __cdecl _wsplitpath(const __wchar_t * _FullPath, __wchar_t * _Drive, __wchar_t * _Dir, __wchar_t * _Filename, __wchar_t * _Ext); 
#line 388
errno_t __cdecl _wsplitpath_s(const __wchar_t * _FullPath, __wchar_t * _Drive, size_t _DriveCount, __wchar_t * _Dir, size_t _DirCount, __wchar_t * _Filename, size_t _FilenameCount, __wchar_t * _Ext, size_t _ExtCount); 
#line 400
extern "C++" {template < size_t _DriveSize, size_t _DirSize, size_t _NameSize, size_t _ExtSize > inline errno_t __cdecl _wsplitpath_s ( wchar_t const * _Path, wchar_t ( & _Drive ) [ _DriveSize ], wchar_t ( & _Dir ) [ _DirSize ], wchar_t ( & _Name ) [ _NameSize ], wchar_t ( & _Ext ) [ _ExtSize ] ) throw ( ) { return _wsplitpath_s ( _Path, _Drive, _DriveSize, _Dir, _DirSize, _Name, _NameSize, _Ext, _ExtSize ); }}
#line 409 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\corecrt_wstdlib.h"
errno_t __cdecl _wdupenv_s(__wchar_t ** _Buffer, size_t * _BufferCount, const __wchar_t * _VarName); 
#line 418
__wchar_t *__cdecl _wgetenv(const __wchar_t * _VarName); 
#line 424
errno_t __cdecl _wgetenv_s(size_t * _RequiredCount, __wchar_t * _Buffer, size_t _BufferCount, const __wchar_t * _VarName); 
#line 431
extern "C++" {template < size_t _Size > inline errno_t __cdecl _wgetenv_s ( size_t * _RequiredCount, wchar_t ( & _Buffer ) [ _Size ], wchar_t const * _VarName ) throw ( ) { return _wgetenv_s ( _RequiredCount, _Buffer, _Size, _VarName ); }}
#line 440 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\corecrt_wstdlib.h"
int __cdecl _wputenv(const __wchar_t * _EnvString); 
#line 445
errno_t __cdecl _wputenv_s(const __wchar_t * _Name, const __wchar_t * _Value); 
#line 450
errno_t __cdecl _wsearchenv_s(const __wchar_t * _Filename, const __wchar_t * _VarName, __wchar_t * _Buffer, size_t _BufferCount); 
#line 457
extern "C++" {template < size_t _Size > inline errno_t __cdecl _wsearchenv_s ( wchar_t const * _Filename, wchar_t const * _VarName, wchar_t ( & _ResultPath ) [ _Size ] ) throw ( ) { return _wsearchenv_s ( _Filename, _VarName, _ResultPath, _Size ); }}
#line 464 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\corecrt_wstdlib.h"
void __cdecl _wsearchenv(const __wchar_t * _Filename, const __wchar_t * _VarName, __wchar_t * _ResultPath); 
#line 471 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\corecrt_wstdlib.h"
int __cdecl _wsystem(const __wchar_t * _Command); 
#line 479 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\corecrt_wstdlib.h"
}__pragma( pack ( pop )) 
#line 481
#pragma warning(pop)
#line 18 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\stdlib.h"
#pragma warning(push)
#pragma warning(disable: 4324 4514 4574 4710 4793 4820 4995 4996 28719 28726 28727 )
#line 22
__pragma( pack ( push, 8 )) extern "C" {
#line 38 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\stdlib.h"
void __cdecl _swab(char * _Buf1, char * _Buf2, int _SizeInBytes); 
#line 56
__declspec(noreturn) void __cdecl exit(int _Code); 
#line 57
__declspec(noreturn) void __cdecl _exit(int _Code); 
#line 58
__declspec(noreturn) void __cdecl _Exit(int _Code); 
#line 59
__declspec(noreturn) void __cdecl quick_exit(int _Code); 
#line 60
__declspec(noreturn) void __cdecl abort(); 
#line 67 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\stdlib.h"
unsigned __cdecl _set_abort_behavior(unsigned _Flags, unsigned _Mask); 
#line 77
typedef int (__cdecl *_onexit_t)(void); 
#line 144 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\stdlib.h"
int __cdecl atexit(void (__cdecl *)(void)); 
#line 145
_onexit_t __cdecl _onexit(_onexit_t _Func); 
#line 148 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\stdlib.h"
int __cdecl at_quick_exit(void (__cdecl *)(void)); 
#line 159
typedef void (__cdecl *_purecall_handler)(void); 
#line 162
typedef void (__cdecl *_invalid_parameter_handler)(const __wchar_t *, const __wchar_t *, const __wchar_t *, unsigned, uintptr_t); 
#line 171
_purecall_handler __cdecl _set_purecall_handler(_purecall_handler _Handler); 
#line 175
_purecall_handler __cdecl _get_purecall_handler(); 
#line 178
_invalid_parameter_handler __cdecl _set_invalid_parameter_handler(_invalid_parameter_handler _Handler); 
#line 182
_invalid_parameter_handler __cdecl _get_invalid_parameter_handler(); 
#line 184
_invalid_parameter_handler __cdecl _set_thread_local_invalid_parameter_handler(_invalid_parameter_handler _Handler); 
#line 188
_invalid_parameter_handler __cdecl _get_thread_local_invalid_parameter_handler(); 
#line 212 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\stdlib.h"
int __cdecl _set_error_mode(int _Mode); 
#line 217
int *__cdecl _errno(); 
#line 220
errno_t __cdecl _set_errno(int _Value); 
#line 221
errno_t __cdecl _get_errno(int * _Value); 
#line 223
unsigned long *__cdecl __doserrno(); 
#line 226
errno_t __cdecl _set_doserrno(unsigned long _Value); 
#line 227
errno_t __cdecl _get_doserrno(unsigned long * _Value); 
#line 230
char **__cdecl __sys_errlist(); 
#line 233
int *__cdecl __sys_nerr(); 
#line 236
void __cdecl perror(const char * _ErrMsg); 
#line 242 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\stdlib.h"
char **__cdecl __p__pgmptr(); 
#line 243
__wchar_t **__cdecl __p__wpgmptr(); 
#line 244
int *__cdecl __p__fmode(); 
#line 259 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\stdlib.h"
errno_t __cdecl _get_pgmptr(char ** _Value); 
#line 262
errno_t __cdecl _get_wpgmptr(__wchar_t ** _Value); 
#line 264
errno_t __cdecl _set_fmode(int _Mode); 
#line 266
errno_t __cdecl _get_fmode(int * _PMode); 
#line 279
typedef 
#line 275
struct _div_t { 
#line 277
int quot; 
#line 278
int rem; 
#line 279
} div_t; 
#line 285
typedef 
#line 281
struct _ldiv_t { 
#line 283
long quot; 
#line 284
long rem; 
#line 285
} ldiv_t; 
#line 291
typedef 
#line 287
struct _lldiv_t { 
#line 289
__int64 quot; 
#line 290
__int64 rem; 
#line 291
} lldiv_t; 
#line 293
int __cdecl abs(int _Number); 
#line 294
long __cdecl labs(long _Number); 
#line 295
__int64 __cdecl llabs(__int64 _Number); 
#line 296
__int64 __cdecl _abs64(__int64 _Number); 
#line 298
unsigned short __cdecl _byteswap_ushort(unsigned short _Number); 
#line 299
unsigned long __cdecl _byteswap_ulong(unsigned long _Number); 
#line 300
unsigned __int64 __cdecl _byteswap_uint64(unsigned __int64 _Number); 
#line 302
div_t __cdecl div(int _Numerator, int _Denominator); 
#line 303
ldiv_t __cdecl ldiv(long _Numerator, long _Denominator); 
#line 304
lldiv_t __cdecl lldiv(__int64 _Numerator, __int64 _Denominator); 
#line 308
#pragma warning(push)
#pragma warning(disable: 6540)
#line 311
unsigned __cdecl _rotl(unsigned _Value, int _Shift); 
#line 317
unsigned long __cdecl _lrotl(unsigned long _Value, int _Shift); 
#line 322
unsigned __int64 __cdecl _rotl64(unsigned __int64 _Value, int _Shift); 
#line 327
unsigned __cdecl _rotr(unsigned _Value, int _Shift); 
#line 333
unsigned long __cdecl _lrotr(unsigned long _Value, int _Shift); 
#line 338
unsigned __int64 __cdecl _rotr64(unsigned __int64 _Value, int _Shift); 
#line 343
#pragma warning(pop)
#line 350
void __cdecl srand(unsigned _Seed); 
#line 352
int __cdecl rand(); 
#line 361 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\stdlib.h"
extern "C++" {
#line 363
inline long abs(const long _X) throw() 
#line 364
{ 
#line 365
return labs(_X); 
#line 366
} 
#line 368
inline __int64 abs(const __int64 _X) throw() 
#line 369
{ 
#line 370
return llabs(_X); 
#line 371
} 
#line 373
inline ldiv_t div(const long _A1, const long _A2) throw() 
#line 374
{ 
#line 375
return ldiv(_A1, _A2); 
#line 376
} 
#line 378
inline lldiv_t div(const __int64 _A1, const __int64 _A2) throw() 
#line 379
{ 
#line 380
return lldiv(_A1, _A2); 
#line 381
} 
#line 382
}
#line 394 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\stdlib.h"
#pragma pack ( push, 4 )
#line 398
typedef 
#line 396
struct { 
#line 397
unsigned char ld[10]; 
#line 398
} _LDOUBLE; 
#pragma pack ( pop )
#line 418 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\stdlib.h"
typedef 
#line 416
struct { 
#line 417
double x; 
#line 418
} _CRT_DOUBLE; 
#line 423
typedef 
#line 421
struct { 
#line 422
float f; 
#line 423
} _CRT_FLOAT; 
#line 432
typedef 
#line 430
struct { 
#line 431
long double x; 
#line 432
} _LONGDOUBLE; 
#line 436
#pragma pack ( push, 4 )
#line 440
typedef 
#line 438
struct { 
#line 439
unsigned char ld12[12]; 
#line 440
} _LDBL12; 
#pragma pack ( pop )
#line 450
double __cdecl atof(const char * _String); 
#line 451
int __cdecl atoi(const char * _String); 
#line 452
long __cdecl atol(const char * _String); 
#line 453
__int64 __cdecl atoll(const char * _String); 
#line 454
__int64 __cdecl _atoi64(const char * _String); 
#line 456
double __cdecl _atof_l(const char * _String, _locale_t _Locale); 
#line 457
int __cdecl _atoi_l(const char * _String, _locale_t _Locale); 
#line 458
long __cdecl _atol_l(const char * _String, _locale_t _Locale); 
#line 459
__int64 __cdecl _atoll_l(const char * _String, _locale_t _Locale); 
#line 460
__int64 __cdecl _atoi64_l(const char * _String, _locale_t _Locale); 
#line 462
int __cdecl _atoflt(_CRT_FLOAT * _Result, const char * _String); 
#line 463
int __cdecl _atodbl(_CRT_DOUBLE * _Result, char * _String); 
#line 464
int __cdecl _atoldbl(_LDOUBLE * _Result, char * _String); 
#line 467
int __cdecl _atoflt_l(_CRT_FLOAT * _Result, const char * _String, _locale_t _Locale); 
#line 474
int __cdecl _atodbl_l(_CRT_DOUBLE * _Result, char * _String, _locale_t _Locale); 
#line 482
int __cdecl _atoldbl_l(_LDOUBLE * _Result, char * _String, _locale_t _Locale); 
#line 489
float __cdecl strtof(const char * _String, char ** _EndPtr); 
#line 495
float __cdecl _strtof_l(const char * _String, char ** _EndPtr, _locale_t _Locale); 
#line 502
double __cdecl strtod(const char * _String, char ** _EndPtr); 
#line 508
double __cdecl _strtod_l(const char * _String, char ** _EndPtr, _locale_t _Locale); 
#line 515
long double __cdecl strtold(const char * _String, char ** _EndPtr); 
#line 521
long double __cdecl _strtold_l(const char * _String, char ** _EndPtr, _locale_t _Locale); 
#line 528
long __cdecl strtol(const char * _String, char ** _EndPtr, int _Radix); 
#line 535
long __cdecl _strtol_l(const char * _String, char ** _EndPtr, int _Radix, _locale_t _Locale); 
#line 543
__int64 __cdecl strtoll(const char * _String, char ** _EndPtr, int _Radix); 
#line 550
__int64 __cdecl _strtoll_l(const char * _String, char ** _EndPtr, int _Radix, _locale_t _Locale); 
#line 558
unsigned long __cdecl strtoul(const char * _String, char ** _EndPtr, int _Radix); 
#line 565
unsigned long __cdecl _strtoul_l(const char * _String, char ** _EndPtr, int _Radix, _locale_t _Locale); 
#line 573
unsigned __int64 __cdecl strtoull(const char * _String, char ** _EndPtr, int _Radix); 
#line 580
unsigned __int64 __cdecl _strtoull_l(const char * _String, char ** _EndPtr, int _Radix, _locale_t _Locale); 
#line 588
__int64 __cdecl _strtoi64(const char * _String, char ** _EndPtr, int _Radix); 
#line 595
__int64 __cdecl _strtoi64_l(const char * _String, char ** _EndPtr, int _Radix, _locale_t _Locale); 
#line 603
unsigned __int64 __cdecl _strtoui64(const char * _String, char ** _EndPtr, int _Radix); 
#line 610
unsigned __int64 __cdecl _strtoui64_l(const char * _String, char ** _EndPtr, int _Radix, _locale_t _Locale); 
#line 626
errno_t __cdecl _itoa_s(int _Value, char * _Buffer, size_t _BufferCount, int _Radix); 
#line 633
extern "C++" {template < size_t _Size > inline errno_t __cdecl _itoa_s ( int _Value, char ( & _Buffer ) [ _Size ], int _Radix ) throw ( ) { return _itoa_s ( _Value, _Buffer, _Size, _Radix ); }}
#line 641 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\stdlib.h"
char *__cdecl _itoa(int _Value, char * _Buffer, int _Radix); 
#line 650 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\stdlib.h"
errno_t __cdecl _ltoa_s(long _Value, char * _Buffer, size_t _BufferCount, int _Radix); 
#line 657
extern "C++" {template < size_t _Size > inline errno_t __cdecl _ltoa_s ( long _Value, char ( & _Buffer ) [ _Size ], int _Radix ) throw ( ) { return _ltoa_s ( _Value, _Buffer, _Size, _Radix ); }}
#line 664 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\stdlib.h"
char *__cdecl _ltoa(long _Value, char * _Buffer, int _Radix); 
#line 673 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\stdlib.h"
errno_t __cdecl _ultoa_s(unsigned long _Value, char * _Buffer, size_t _BufferCount, int _Radix); 
#line 680
extern "C++" {template < size_t _Size > inline errno_t __cdecl _ultoa_s ( unsigned long _Value, char ( & _Buffer ) [ _Size ], int _Radix ) throw ( ) { return _ultoa_s ( _Value, _Buffer, _Size, _Radix ); }}
#line 687 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\stdlib.h"
char *__cdecl _ultoa(unsigned long _Value, char * _Buffer, int _Radix); 
#line 696 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\stdlib.h"
errno_t __cdecl _i64toa_s(__int64 _Value, char * _Buffer, size_t _BufferCount, int _Radix); 
#line 705
char *__cdecl _i64toa(__int64 _Value, char * _Buffer, int _Radix); 
#line 713
errno_t __cdecl _ui64toa_s(unsigned __int64 _Value, char * _Buffer, size_t _BufferCount, int _Radix); 
#line 721
char *__cdecl _ui64toa(unsigned __int64 _Value, char * _Buffer, int _Radix); 
#line 741
errno_t __cdecl _ecvt_s(char * _Buffer, size_t _BufferCount, double _Value, int _DigitCount, int * _PtDec, int * _PtSign); 
#line 750
extern "C++" {template < size_t _Size > inline errno_t __cdecl _ecvt_s ( char ( & _Buffer ) [ _Size ], double _Value, int _DigitCount, int * _PtDec, int * _PtSign ) throw ( ) { return _ecvt_s ( _Buffer, _Size, _Value, _DigitCount, _PtDec, _PtSign ); }}
#line 760 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\stdlib.h"
char *__cdecl _ecvt(double _Value, int _DigitCount, int * _PtDec, int * _PtSign); 
#line 769
errno_t __cdecl _fcvt_s(char * _Buffer, size_t _BufferCount, double _Value, int _FractionalDigitCount, int * _PtDec, int * _PtSign); 
#line 778
extern "C++" {template < size_t _Size > inline errno_t __cdecl _fcvt_s ( char ( & _Buffer ) [ _Size ], double _Value, int _FractionalDigitCount, int * _PtDec, int * _PtSign ) throw ( ) { return _fcvt_s ( _Buffer, _Size, _Value, _FractionalDigitCount, _PtDec, _PtSign ); }}
#line 790 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\stdlib.h"
char *__cdecl _fcvt(double _Value, int _FractionalDigitCount, int * _PtDec, int * _PtSign); 
#line 798
errno_t __cdecl _gcvt_s(char * _Buffer, size_t _BufferCount, double _Value, int _DigitCount); 
#line 805
extern "C++" {template < size_t _Size > inline errno_t __cdecl _gcvt_s ( char ( & _Buffer ) [ _Size ], double _Value, int _DigitCount ) throw ( ) { return _gcvt_s ( _Buffer, _Size, _Value, _DigitCount ); }}
#line 814 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\stdlib.h"
char *__cdecl _gcvt(double _Value, int _DigitCount, char * _Buffer); 
#line 843 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\stdlib.h"
int __cdecl ___mb_cur_max_func(); 
#line 846
int __cdecl ___mb_cur_max_l_func(_locale_t _Locale); 
#line 852 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\stdlib.h"
int __cdecl mblen(const char * _Ch, size_t _MaxCount); 
#line 858
int __cdecl _mblen_l(const char * _Ch, size_t _MaxCount, _locale_t _Locale); 
#line 866
size_t __cdecl _mbstrlen(const char * _String); 
#line 872
size_t __cdecl _mbstrlen_l(const char * _String, _locale_t _Locale); 
#line 879
size_t __cdecl _mbstrnlen(const char * _String, size_t _MaxCount); 
#line 886
size_t __cdecl _mbstrnlen_l(const char * _String, size_t _MaxCount, _locale_t _Locale); 
#line 893
int __cdecl mbtowc(__wchar_t * _DstCh, const char * _SrcCh, size_t _SrcSizeInBytes); 
#line 900
int __cdecl _mbtowc_l(__wchar_t * _DstCh, const char * _SrcCh, size_t _SrcSizeInBytes, _locale_t _Locale); 
#line 908
errno_t __cdecl mbstowcs_s(size_t * _PtNumOfCharConverted, __wchar_t * _DstBuf, size_t _SizeInWords, const char * _SrcBuf, size_t _MaxCount); 
#line 916
extern "C++" {template < size_t _Size > inline errno_t __cdecl mbstowcs_s ( size_t * _PtNumOfCharConverted, wchar_t ( & _Dest ) [ _Size ], char const * _Source, size_t _MaxCount ) throw ( ) { return mbstowcs_s ( _PtNumOfCharConverted, _Dest, _Size, _Source, _MaxCount ); }}
#line 924 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\stdlib.h"
size_t __cdecl mbstowcs(__wchar_t * _Dest, const char * _Source, size_t _MaxCount); 
#line 932 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\stdlib.h"
errno_t __cdecl _mbstowcs_s_l(size_t * _PtNumOfCharConverted, __wchar_t * _DstBuf, size_t _SizeInWords, const char * _SrcBuf, size_t _MaxCount, _locale_t _Locale); 
#line 941
extern "C++" {template < size_t _Size > inline errno_t __cdecl _mbstowcs_s_l ( size_t * _PtNumOfCharConverted, wchar_t ( & _Dest ) [ _Size ], char const * _Source, size_t _MaxCount, _locale_t _Locale ) throw ( ) { return _mbstowcs_s_l ( _PtNumOfCharConverted, _Dest, _Size, _Source, _MaxCount, _Locale ); }}
#line 950 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\stdlib.h"
size_t __cdecl _mbstowcs_l(__wchar_t * _Dest, const char * _Source, size_t _MaxCount, _locale_t _Locale); 
#line 963 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\stdlib.h"
int __cdecl wctomb(char * _MbCh, __wchar_t _WCh); 
#line 969
int __cdecl _wctomb_l(char * _MbCh, __wchar_t _WCh, _locale_t _Locale); 
#line 978
errno_t __cdecl wctomb_s(int * _SizeConverted, char * _MbCh, rsize_t _SizeInBytes, __wchar_t _WCh); 
#line 988 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\stdlib.h"
errno_t __cdecl _wctomb_s_l(int * _SizeConverted, char * _MbCh, size_t _SizeInBytes, __wchar_t _WCh, _locale_t _Locale); 
#line 996
errno_t __cdecl wcstombs_s(size_t * _PtNumOfCharConverted, char * _Dst, size_t _DstSizeInBytes, const __wchar_t * _Src, size_t _MaxCountInBytes); 
#line 1004
extern "C++" {template < size_t _Size > inline errno_t __cdecl wcstombs_s ( size_t * _PtNumOfCharConverted, char ( & _Dest ) [ _Size ], wchar_t const * _Source, size_t _MaxCount ) throw ( ) { return wcstombs_s ( _PtNumOfCharConverted, _Dest, _Size, _Source, _MaxCount ); }}
#line 1012 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\stdlib.h"
size_t __cdecl wcstombs(char * _Dest, const __wchar_t * _Source, size_t _MaxCount); 
#line 1020 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\stdlib.h"
errno_t __cdecl _wcstombs_s_l(size_t * _PtNumOfCharConverted, char * _Dst, size_t _DstSizeInBytes, const __wchar_t * _Src, size_t _MaxCountInBytes, _locale_t _Locale); 
#line 1029
extern "C++" {template < size_t _Size > inline errno_t __cdecl _wcstombs_s_l ( size_t * _PtNumOfCharConverted, char ( & _Dest ) [ _Size ], wchar_t const * _Source, size_t _MaxCount, _locale_t _Locale ) throw ( ) { return _wcstombs_s_l ( _PtNumOfCharConverted, _Dest, _Size, _Source, _MaxCount, _Locale ); }}
#line 1038 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\stdlib.h"
size_t __cdecl _wcstombs_l(char * _Dest, const __wchar_t * _Source, size_t _MaxCount, _locale_t _Locale); 
#line 1068 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\stdlib.h"
__declspec(allocator) char *__cdecl _fullpath(char * _Buffer, const char * _Path, size_t _BufferCount); 
#line 1077
errno_t __cdecl _makepath_s(char * _Buffer, size_t _BufferCount, const char * _Drive, const char * _Dir, const char * _Filename, const char * _Ext); 
#line 1086
extern "C++" {template < size_t _Size > inline errno_t __cdecl _makepath_s ( char ( & _Buffer ) [ _Size ], char const * _Drive, char const * _Dir, char const * _Filename, char const * _Ext ) throw ( ) { return _makepath_s ( _Buffer, _Size, _Drive, _Dir, _Filename, _Ext ); }}
#line 1095 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\stdlib.h"
void __cdecl _makepath(char * _Buffer, const char * _Drive, const char * _Dir, const char * _Filename, const char * _Ext); 
#line 1105 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\stdlib.h"
void __cdecl _splitpath(const char * _FullPath, char * _Drive, char * _Dir, char * _Filename, char * _Ext); 
#line 1114
errno_t __cdecl _splitpath_s(const char * _FullPath, char * _Drive, size_t _DriveCount, char * _Dir, size_t _DirCount, char * _Filename, size_t _FilenameCount, char * _Ext, size_t _ExtCount); 
#line 1126
extern "C++" {template < size_t _DriveSize, size_t _DirSize, size_t _NameSize, size_t _ExtSize > inline errno_t __cdecl _splitpath_s ( char const * _Dest, char ( & _Drive ) [ _DriveSize ], char ( & _Dir ) [ _DirSize ], char ( & _Name ) [ _NameSize ], char ( & _Ext ) [ _ExtSize ] ) throw ( ) { return _splitpath_s ( _Dest, _Drive, _DriveSize, _Dir, _DirSize, _Name, _NameSize, _Ext, _ExtSize ); }}
#line 1132
errno_t __cdecl getenv_s(size_t * _RequiredCount, char * _Buffer, rsize_t _BufferCount, const char * _VarName); 
#line 1144 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\stdlib.h"
int *__cdecl __p___argc(); 
#line 1145
char ***__cdecl __p___argv(); 
#line 1146
__wchar_t ***__cdecl __p___wargv(); 
#line 1158 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\stdlib.h"
char ***__cdecl __p__environ(); 
#line 1159
__wchar_t ***__cdecl __p__wenviron(); 
#line 1184 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\stdlib.h"
char *__cdecl getenv(const char * _VarName); 
#line 1188
extern "C++" {template < size_t _Size > inline errno_t __cdecl getenv_s ( size_t * _RequiredCount, char ( & _Buffer ) [ _Size ], char const * _VarName ) throw ( ) { return getenv_s ( _RequiredCount, _Buffer, _Size, _VarName ); }}
#line 1201 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\stdlib.h"
errno_t __cdecl _dupenv_s(char ** _Buffer, size_t * _BufferCount, const char * _VarName); 
#line 1211 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\stdlib.h"
int __cdecl system(const char * _Command); 
#line 1217
#pragma warning(push)
#pragma warning(disable: 6540)
#line 1221
int __cdecl _putenv(const char * _EnvString); 
#line 1226
errno_t __cdecl _putenv_s(const char * _Name, const char * _Value); 
#line 1231
#pragma warning(pop)
#line 1233
errno_t __cdecl _searchenv_s(const char * _Filename, const char * _VarName, char * _Buffer, size_t _BufferCount); 
#line 1240
extern "C++" {template < size_t _Size > inline errno_t __cdecl _searchenv_s ( char const * _Filename, char const * _VarName, char ( & _Buffer ) [ _Size ] ) throw ( ) { return _searchenv_s ( _Filename, _VarName, _Buffer, _Size ); }}
#line 1247 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\stdlib.h"
void __cdecl _searchenv(const char * _Filename, const char * _VarName, char * _Buffer); 
#line 1255 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\stdlib.h"
__declspec(deprecated("This function or variable has been superceded by newer library or operating system functionality. Consider using SetErrorMode in" "stead. See online help for details.")) void __cdecl 
#line 1256
_seterrormode(int _Mode); 
#line 1260
__declspec(deprecated("This function or variable has been superceded by newer library or operating system functionality. Consider using Beep instead. S" "ee online help for details.")) void __cdecl 
#line 1261
_beep(unsigned _Frequency, unsigned _Duration); 
#line 1266
__declspec(deprecated("This function or variable has been superceded by newer library or operating system functionality. Consider using Sleep instead. " "See online help for details.")) void __cdecl 
#line 1267
_sleep(unsigned long _Duration); 
#line 1289 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\stdlib.h"
#pragma warning(push)
#pragma warning(disable: 4141)
#line 1293
char *__cdecl ecvt(double _Value, int _DigitCount, int * _PtDec, int * _PtSign); 
#line 1301
char *__cdecl fcvt(double _Value, int _FractionalDigitCount, int * _PtDec, int * _PtSign); 
#line 1309
char *__cdecl gcvt(double _Value, int _DigitCount, char * _DstBuf); 
#line 1316
char *__cdecl itoa(int _Value, char * _Buffer, int _Radix); 
#line 1323
char *__cdecl ltoa(long _Value, char * _Buffer, int _Radix); 
#line 1331
void __cdecl swab(char * _Buf1, char * _Buf2, int _SizeInBytes); 
#line 1338
char *__cdecl ultoa(unsigned long _Value, char * _Buffer, int _Radix); 
#line 1347
int __cdecl putenv(const char * _EnvString); 
#line 1351
#pragma warning(pop)
#line 1353
_onexit_t __cdecl onexit(_onexit_t _Func); 
#line 1359 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\stdlib.h"
}__pragma( pack ( pop )) 
#line 1361
#pragma warning(pop)
#line 12 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Enterprise\\VC\\Tools\\MSVC\\14.37.32822\\include\\vcruntime_new.h"
#pragma warning(push)
#pragma warning(disable: 4514 4820 )
#pragma warning(disable: 4985)
#line 17
extern "C++" {
#line 19
#pragma pack ( push, 8 )
#line 33
namespace std { 
#line 35
struct nothrow_t { 
#line 36
explicit nothrow_t() = default;
#line 37
}; 
#line 42
extern const nothrow_t nothrow; 
#line 44 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Enterprise\\VC\\Tools\\MSVC\\14.37.32822\\include\\vcruntime_new.h"
}
#line 47 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Enterprise\\VC\\Tools\\MSVC\\14.37.32822\\include\\vcruntime_new.h"
[[nodiscard]] __declspec(allocator) void *__cdecl 
#line 48
operator new(size_t _Size); 
#line 52
[[nodiscard]] __declspec(allocator) void *__cdecl 
#line 53
operator new(size_t _Size, const std::nothrow_t &) noexcept; 
#line 58
[[nodiscard]] __declspec(allocator) void *__cdecl 
#line 59
operator new[](size_t _Size); 
#line 63
[[nodiscard]] __declspec(allocator) void *__cdecl 
#line 64
operator new[](size_t _Size, const std::nothrow_t &) noexcept; 
#line 69
void __cdecl operator delete(void * _Block) noexcept; 
#line 73
void __cdecl operator delete(void * _Block, const std::nothrow_t &) noexcept; 
#line 78
void __cdecl operator delete[](void * _Block) noexcept; 
#line 82
void __cdecl operator delete[](void * _Block, const std::nothrow_t &) noexcept; 
#line 87
void __cdecl operator delete(void * _Block, size_t _Size) noexcept; 
#line 92
void __cdecl operator delete[](void * _Block, size_t _Size) noexcept; 
#line 160
#pragma warning(push)
#pragma warning(disable: 4577)
#pragma warning(disable: 4514)
#line 165
[[nodiscard]] [[msvc::constexpr]] inline void *__cdecl 
#line 166
operator new(size_t _Size, void *
#line 167
_Where) noexcept 
#line 168
{ 
#line 169
(void)_Size; 
#line 170
return _Where; 
#line 171
} 
#line 173
inline void __cdecl operator delete(void *, void *) noexcept 
#line 174
{ 
#line 176
} 
#line 181 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Enterprise\\VC\\Tools\\MSVC\\14.37.32822\\include\\vcruntime_new.h"
[[nodiscard]] inline void *__cdecl 
#line 182
operator new[](size_t _Size, void *
#line 183
_Where) noexcept 
#line 184
{ 
#line 185
(void)_Size; 
#line 186
return _Where; 
#line 187
} 
#line 189
inline void __cdecl operator delete[](void *, void *) noexcept 
#line 190
{ 
#line 191
} 
#line 199 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Enterprise\\VC\\Tools\\MSVC\\14.37.32822\\include\\vcruntime_new.h"
}
#line 193
#pragma warning(pop)
#line 197
#pragma pack ( pop )
#line 202 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Enterprise\\VC\\Tools\\MSVC\\14.37.32822\\include\\vcruntime_new.h"
#pragma warning(pop)
#line 12 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Enterprise\\VC\\Tools\\MSVC\\14.37.32822\\include\\vcruntime_new_debug.h"
#pragma warning(push)
#pragma warning(disable: 4514 4820 )
#line 16
extern "C++" {
#line 18
#pragma pack ( push, 8 )
#line 25
[[nodiscard]] 
#line 26
__declspec(allocator) void *__cdecl operator new(size_t _Size, int _BlockUse, const char * _FileName, int _LineNumber); 
#line 33
[[nodiscard]] 
#line 34
__declspec(allocator) void *__cdecl operator new[](size_t _Size, int _BlockUse, const char * _FileName, int _LineNumber); 
#line 41
void __cdecl operator delete(void * _Block, int _BlockUse, const char * _FileName, int _LineNumber) noexcept; 
#line 48
void __cdecl operator delete[](void * _Block, int _BlockUse, const char * _FileName, int _LineNumber) noexcept; 
#line 61 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Enterprise\\VC\\Tools\\MSVC\\14.37.32822\\include\\vcruntime_new_debug.h"
}
#line 59
#pragma pack ( pop )
#line 64 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Enterprise\\VC\\Tools\\MSVC\\14.37.32822\\include\\vcruntime_new_debug.h"
#pragma warning(pop)
#line 15 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\crtdbg.h"
#pragma warning(push)
#pragma warning(disable: 4324 4514 4574 4710 4793 4820 4995 4996 28719 28726 28727 )
#line 19
__pragma( pack ( push, 8 )) extern "C" {
#line 23
typedef void *_HFILE; 
#line 49
typedef int (__cdecl *_CRT_REPORT_HOOK)(int, char *, int *); 
#line 50
typedef int (__cdecl *_CRT_REPORT_HOOKW)(int, __wchar_t *, int *); 
#line 56
typedef int (__cdecl *_CRT_ALLOC_HOOK)(int, void *, size_t, int, long, const unsigned char *, int); 
#line 112
typedef void (__cdecl *_CRT_DUMP_CLIENT)(void *, size_t); 
#line 118
struct _CrtMemBlockHeader; 
#line 127
typedef 
#line 120
struct _CrtMemState { 
#line 122
_CrtMemBlockHeader *pBlockHeader; 
#line 123
size_t lCounts[5]; 
#line 124
size_t lSizes[5]; 
#line 125
size_t lHighWaterCount; 
#line 126
size_t lTotalCount; 
#line 127
} _CrtMemState; 
#line 813 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.19041.0\\ucrt\\crtdbg.h"
}__pragma( pack ( pop )) 
#line 815
#pragma warning(pop)
#line 24 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Enterprise\\VC\\Tools\\MSVC\\14.37.32822\\include\\yvals.h"
#pragma pack ( push, 8 )
#pragma warning(push, 3)
#pragma warning(disable : 4180 4412 4455 4494 4514 4574 4582 4583 4587 4588 4619 4623 4625 4626 4643 4648 4702 4793 4820 4988 5026 5027 5045 6294 4984 5053 )
#line 142 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Enterprise\\VC\\Tools\\MSVC\\14.37.32822\\include\\yvals.h"
#pragma detect_mismatch("_MSC_VER", "1900")
#line 146 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Enterprise\\VC\\Tools\\MSVC\\14.37.32822\\include\\yvals.h"
#pragma detect_mismatch("_ITERATOR_DEBUG_LEVEL", "0")
#line 151 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Enterprise\\VC\\Tools\\MSVC\\14.37.32822\\include\\yvals.h"
#pragma detect_mismatch("RuntimeLibrary", "MT_StaticRelease")
#line 49 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Enterprise\\VC\\Tools\\MSVC\\14.37.32822\\include\\use_ansi.h"
#pragma comment(lib, "libcpmt")
#line 349 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Enterprise\\VC\\Tools\\MSVC\\14.37.32822\\include\\yvals.h"
namespace std { 
#line 350
enum _Uninitialized { 
#line 351
_Noinit
#line 352
}; 
#line 354
class _Lockit { 
#line 370
public: __thiscall _Lockit() noexcept; 
#line 371
explicit __thiscall _Lockit(int) noexcept; 
#line 372
__thiscall ~_Lockit() noexcept; 
#line 375 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Enterprise\\VC\\Tools\\MSVC\\14.37.32822\\include\\yvals.h"
static void __cdecl _Lockit_ctor(int) noexcept; 
#line 376
static void __cdecl _Lockit_dtor(int) noexcept; 
#line 379
private: static void __cdecl _Lockit_ctor(_Lockit *) noexcept; 
#line 380
static void __cdecl _Lockit_ctor(_Lockit *, int) noexcept; 
#line 381
static void __cdecl _Lockit_dtor(_Lockit *) noexcept; 
#line 384
public: _Lockit(const _Lockit &) = delete;
#line 385
_Lockit &operator=(const _Lockit &) = delete;
#line 388
private: int _Locktype; 
#line 389
}; 
#line 508 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Enterprise\\VC\\Tools\\MSVC\\14.37.32822\\include\\yvals.h"
}
#line 512
#pragma warning(pop)
#pragma pack ( pop )
#line 15 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Enterprise\\VC\\Tools\\MSVC\\14.37.32822\\include\\cstdlib"
#pragma pack ( push, 8 )
#pragma warning(push, 3)
#pragma warning(disable : 4180 4412 4455 4494 4514 4574 4582 4583 4587 4588 4619 4623 4625 4626 4643 4648 4702 4793 4820 4988 5026 5027 5045 6294 4984 5053 )
#line 23
[[nodiscard]] inline double abs(double _Xx) noexcept { 
#line 24
return ::fabs(_Xx); 
#line 25
} 
#line 27
[[nodiscard]] inline float abs(float _Xx) noexcept { 
#line 28
return ::fabsf(_Xx); 
#line 29
} 
#line 31
[[nodiscard]] inline long double abs(long double _Xx) noexcept { 
#line 32
return ::fabsl(_Xx); 
#line 33
} 
#line 35
namespace std { 
#line 36
using ::size_t;
#line 37
using ::div_t;
#line 38
using ::ldiv_t;
#line 39
using ::abort;
#line 40
using ::abs;
#line 41
using ::atexit;
#line 42
using ::atof;
#line 43
using ::atoi;
#line 44
using ::atol;
#line 45
using ::bsearch;
#line 46
using ::calloc;
#line 47
using ::div;
#line 48
using ::exit;
#line 49
using ::free;
#line 50
using ::labs;
#line 51
using ::ldiv;
#line 52
using ::malloc;
#line 53
using ::mblen;
#line 54
using ::mbstowcs;
#line 55
using ::mbtowc;
#line 56
using ::qsort;
#line 57
using ::rand;
#line 58
using ::realloc;
#line 59
using ::srand;
#line 60
using ::strtod;
#line 61
using ::strtol;
#line 62
using ::strtoul;
#line 63
using ::wcstombs;
#line 64
using ::wctomb;
#line 66
using ::lldiv_t;
#line 68
using ::getenv;
#line 69
using ::system;
#line 71
using ::atoll;
#line 72
using ::llabs;
#line 73
using ::lldiv;
#line 74
using ::strtof;
#line 75
using ::strtold;
#line 76
using ::strtoll;
#line 77
using ::strtoull;
#line 79
using ::_Exit;
#line 80
using ::at_quick_exit;
#line 81
using ::quick_exit;
#line 82
}
#line 86
#pragma warning(pop)
#pragma pack ( pop )
#line 12 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Enterprise\\VC\\Tools\\MSVC\\14.37.32822\\include\\xtr1common"
#pragma pack ( push, 8 )
#pragma warning(push, 3)
#pragma warning(disable : 4180 4412 4455 4494 4514 4574 4582 4583 4587 4588 4619 4623 4625 4626 4643 4648 4702 4793 4820 4988 5026 5027 5045 6294 4984 5053 )
#line 19
namespace std { 
#line 20
template < class _Ty, _Ty _Val >
struct integral_constant {
    static constexpr _Ty value = _Val;

    using value_type = _Ty;
    using type = integral_constant;

    constexpr operator value_type ( ) const noexcept {
        return value;
    }

    [ [ nodiscard ] ] constexpr value_type operator ( ) ( ) const noexcept {
        return value;
    }
};
#line 36
template< bool _Val> using bool_constant = integral_constant< bool, _Val> ; 
#line 39
using true_type = bool_constant< true> ; 
#line 40
using false_type = bool_constant< false> ; 
#line 42
template< bool _Test, class _Ty = void> 
#line 43
struct enable_if { }; 
#line 45
template< class _Ty> 
#line 46
struct enable_if< true, _Ty>  { 
#line 47
using type = _Ty; 
#line 48
}; 
#line 50
template< bool _Test, class _Ty = void> using enable_if_t = typename enable_if< _Test, _Ty> ::type; 
#line 53
template< bool _Test, class _Ty1, class _Ty2> 
#line 54
struct conditional { 
#line 55
using type = _Ty1; 
#line 56
}; 
#line 58
template< class _Ty1, class _Ty2> 
#line 59
struct conditional< false, _Ty1, _Ty2>  { 
#line 60
using type = _Ty2; 
#line 61
}; 
#line 63
template< bool _Test, class _Ty1, class _Ty2> using conditional_t = typename conditional< _Test, _Ty1, _Ty2> ::type; 
#line 73
template< class , class > constexpr bool 
#line 74
is_same_v = false; 
#line 75
template< class _Ty> constexpr bool 
#line 76
is_same_v< _Ty, _Ty>  = true; 
#line 78
template< class _Ty1, class _Ty2> 
#line 79
struct is_same : public bool_constant< is_same_v< _Ty1, _Ty2> >  { }; 
#line 82 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Enterprise\\VC\\Tools\\MSVC\\14.37.32822\\include\\xtr1common"
template< class _Ty> 
#line 83
struct remove_const { 
#line 84
using type = _Ty; 
#line 85
}; 
#line 87
template< class _Ty> 
#line 88
struct remove_const< const _Ty>  { 
#line 89
using type = _Ty; 
#line 90
}; 
#line 92
template< class _Ty> using remove_const_t = typename remove_const< _Ty> ::type; 
#line 95
template< class _Ty> 
#line 96
struct remove_volatile { 
#line 97
using type = _Ty; 
#line 98
}; 
#line 100
template< class _Ty> 
#line 101
struct remove_volatile< volatile _Ty>  { 
#line 102
using type = _Ty; 
#line 103
}; 
#line 105
template< class _Ty> using remove_volatile_t = typename remove_volatile< _Ty> ::type; 
#line 108
template< class _Ty> 
#line 109
struct remove_cv { 
#line 110
using type = _Ty; 
#line 112
template< template< class >  class _Fn> using _Apply = _Fn< _Ty> ; 
#line 114
}; 
#line 116
template< class _Ty> 
#line 117
struct remove_cv< const _Ty>  { 
#line 118
using type = _Ty; 
#line 120
template< template< class >  class _Fn> using _Apply = const _Fn< _Ty> ; 
#line 122
}; 
#line 124
template< class _Ty> 
#line 125
struct remove_cv< volatile _Ty>  { 
#line 126
using type = _Ty; 
#line 128
template< template< class >  class _Fn> using _Apply = volatile _Fn< _Ty> ; 
#line 130
}; 
#line 132
template< class _Ty> 
#line 133
struct remove_cv< const volatile _Ty>  { 
#line 134
using type = _Ty; 
#line 136
template< template< class >  class _Fn> using _Apply = const volatile _Fn< _Ty> ; 
#line 138
}; 
#line 140
template< class _Ty> using remove_cv_t = typename remove_cv< _Ty> ::type; 
#line 143
template< bool _First_value, class _First, class ..._Rest> 
#line 144
struct _Disjunction { 
#line 145
using type = _First; 
#line 146
}; 
#line 148
template< class _False, class _Next, class ..._Rest> 
#line 149
struct _Disjunction< false, _False, _Next, _Rest...>  { 
#line 150
using type = typename std::_Disjunction< _Next::value, _Next, _Rest...> ::type; 
#line 151
}; 
#line 153
template< class ..._Traits> 
#line 154
struct disjunction : public false_type { }; 
#line 156
template< class _First, class ..._Rest> 
#line 157
struct disjunction< _First, _Rest...>  : public _Disjunction< _First::value, _First, _Rest...> ::type { 
#line 159
}; 
#line 161
template< class ..._Traits> constexpr bool 
#line 162
disjunction_v = (disjunction< _Traits...> ::value); 
#line 164
template< class _Ty, class ..._Types> constexpr bool 
#line 165
_Is_any_of_v = disjunction_v< is_same< _Ty, _Types> ...> ; 
#line 168
[[nodiscard]] constexpr bool _Is_constant_evaluated() noexcept { 
#line 169
return __builtin_is_constant_evaluated(); 
#line 170
} 
#line 178 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Enterprise\\VC\\Tools\\MSVC\\14.37.32822\\include\\xtr1common"
template< class _Ty> constexpr bool 
#line 179
is_integral_v = _Is_any_of_v< remove_cv_t< _Ty> , bool, char, signed char, unsigned char, __wchar_t, char16_t, char32_t, short, unsigned short, int, unsigned, long, unsigned long, __int64, unsigned __int64> ; 
#line 186
template< class _Ty> 
#line 187
struct is_integral : public bool_constant< is_integral_v< _Ty> >  { }; 
#line 189
template< class _Ty> constexpr bool 
#line 190
is_floating_point_v = _Is_any_of_v< remove_cv_t< _Ty> , float, double, long double> ; 
#line 192
template< class _Ty> 
#line 193
struct is_floating_point : public bool_constant< is_floating_point_v< _Ty> >  { }; 
#line 195
template< class _Ty> constexpr bool 
#line 196
is_arithmetic_v = is_integral_v< _Ty>  || is_floating_point_v< _Ty> ; 
#line 199
template< class _Ty> 
#line 200
struct is_arithmetic : public bool_constant< is_arithmetic_v< _Ty> >  { }; 
#line 202
template< class _Ty> 
#line 203
struct remove_reference { 
#line 204
using type = _Ty; 
#line 205
using _Const_thru_ref_type = const _Ty; 
#line 206
}; 
#line 208
template< class _Ty> 
#line 209
struct remove_reference< _Ty &>  { 
#line 210
using type = _Ty; 
#line 211
using _Const_thru_ref_type = const _Ty &; 
#line 212
}; 
#line 214
template< class _Ty> 
#line 215
struct remove_reference< _Ty &&>  { 
#line 216
using type = _Ty; 
#line 217
using _Const_thru_ref_type = const _Ty &&; 
#line 218
}; 
#line 220
template< class _Ty> using remove_reference_t = typename remove_reference< _Ty> ::type; 
#line 223
template< class _Ty> using _Const_thru_ref = typename remove_reference< _Ty> ::_Const_thru_ref_type; 
#line 226
template< class _Ty> using _Remove_cvref_t = remove_cv_t< remove_reference_t< _Ty> > ; 
#line 239 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Enterprise\\VC\\Tools\\MSVC\\14.37.32822\\include\\xtr1common"
}
#line 242
#pragma warning(pop)
#pragma pack ( pop )
#line 30 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Enterprise\\VC\\Tools\\MSVC\\14.37.32822\\include\\cmath"
#pragma pack ( push, 8 )
#pragma warning(push, 3)
#pragma warning(disable : 4180 4412 4455 4494 4514 4574 4582 4583 4587 4588 4619 4623 4625 4626 4643 4648 4702 4793 4820 4988 5026 5027 5045 6294 4984 5053 )
#line 37
[[nodiscard]] inline float acos(float _Xx) noexcept { 
#line 38
return ::acosf(_Xx); 
#line 39
} 
#line 41
[[nodiscard]] inline float acosh(float _Xx) noexcept { 
#line 42
return ::acoshf(_Xx); 
#line 43
} 
#line 45
[[nodiscard]] inline float asin(float _Xx) noexcept { 
#line 46
return ::asinf(_Xx); 
#line 47
} 
#line 49
[[nodiscard]] inline float asinh(float _Xx) noexcept { 
#line 50
return ::asinhf(_Xx); 
#line 51
} 
#line 53
[[nodiscard]] inline float atan(float _Xx) noexcept { 
#line 54
return ::atanf(_Xx); 
#line 55
} 
#line 57
[[nodiscard]] inline float atanh(float _Xx) noexcept { 
#line 58
return ::atanhf(_Xx); 
#line 59
} 
#line 61
[[nodiscard]] inline float atan2(float _Yx, float _Xx) noexcept { 
#line 62
return ::atan2f(_Yx, _Xx); 
#line 63
} 
#line 65
[[nodiscard]] inline float cbrt(float _Xx) noexcept { 
#line 66
return ::cbrtf(_Xx); 
#line 67
} 
#line 69
[[nodiscard]] inline float ceil(float _Xx) noexcept { 
#line 75 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Enterprise\\VC\\Tools\\MSVC\\14.37.32822\\include\\cmath"
return ::ceilf(_Xx); 
#line 77 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Enterprise\\VC\\Tools\\MSVC\\14.37.32822\\include\\cmath"
} 
#line 79
[[nodiscard]] inline float copysign(float _Number, float _Sign) noexcept { 
#line 85 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Enterprise\\VC\\Tools\\MSVC\\14.37.32822\\include\\cmath"
return ::copysignf(_Number, _Sign); 
#line 87 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Enterprise\\VC\\Tools\\MSVC\\14.37.32822\\include\\cmath"
} 
#line 89
[[nodiscard]] inline float cos(float _Xx) noexcept { 
#line 90
return ::cosf(_Xx); 
#line 91
} 
#line 93
[[nodiscard]] inline float cosh(float _Xx) noexcept { 
#line 94
return ::coshf(_Xx); 
#line 95
} 
#line 97
[[nodiscard]] inline float erf(float _Xx) noexcept { 
#line 98
return ::erff(_Xx); 
#line 99
} 
#line 101
[[nodiscard]] inline float erfc(float _Xx) noexcept { 
#line 102
return ::erfcf(_Xx); 
#line 103
} 
#line 105
[[nodiscard]] inline float exp(float _Xx) noexcept { 
#line 106
return ::expf(_Xx); 
#line 107
} 
#line 109
[[nodiscard]] inline float exp2(float _Xx) noexcept { 
#line 110
return ::exp2f(_Xx); 
#line 111
} 
#line 113
[[nodiscard]] inline float expm1(float _Xx) noexcept { 
#line 114
return ::expm1f(_Xx); 
#line 115
} 
#line 117
[[nodiscard]] inline float fabs(float _Xx) noexcept { 
#line 118
return ::fabsf(_Xx); 
#line 119
} 
#line 121
[[nodiscard]] inline float fdim(float _Xx, float _Yx) noexcept { 
#line 122
return ::fdimf(_Xx, _Yx); 
#line 123
} 
#line 125
[[nodiscard]] inline float floor(float _Xx) noexcept { 
#line 131 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Enterprise\\VC\\Tools\\MSVC\\14.37.32822\\include\\cmath"
return ::floorf(_Xx); 
#line 133 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Enterprise\\VC\\Tools\\MSVC\\14.37.32822\\include\\cmath"
} 
#line 135
[[nodiscard]] inline float fma(float _Xx, float _Yx, float _Zx) noexcept { 
#line 136
return ::fmaf(_Xx, _Yx, _Zx); 
#line 137
} 
#line 139
[[nodiscard]] inline float fmax(float _Xx, float _Yx) noexcept { 
#line 140
return ::fmaxf(_Xx, _Yx); 
#line 141
} 
#line 143
[[nodiscard]] inline float fmin(float _Xx, float _Yx) noexcept { 
#line 144
return ::fminf(_Xx, _Yx); 
#line 145
} 
#line 147
[[nodiscard]] inline float fmod(float _Xx, float _Yx) noexcept { 
#line 148
return ::fmodf(_Xx, _Yx); 
#line 149
} 
#line 151
inline float frexp(float _Xx, int *_Yx) noexcept { 
#line 152
return ::frexpf(_Xx, _Yx); 
#line 153
} 
#line 155
[[nodiscard]] inline float hypot(float _Xx, float _Yx) noexcept { 
#line 156
return ::hypotf(_Xx, _Yx); 
#line 157
} 
#line 159
[[nodiscard]] inline int ilogb(float _Xx) noexcept { 
#line 160
return ::ilogbf(_Xx); 
#line 161
} 
#line 163
[[nodiscard]] inline float ldexp(float _Xx, int _Yx) noexcept { 
#line 164
return ::ldexpf(_Xx, _Yx); 
#line 165
} 
#line 167
[[nodiscard]] inline float lgamma(float _Xx) noexcept { 
#line 168
return ::lgammaf(_Xx); 
#line 169
} 
#line 171
[[nodiscard]] inline __int64 llrint(float _Xx) noexcept { 
#line 172
return ::llrintf(_Xx); 
#line 173
} 
#line 175
[[nodiscard]] inline __int64 llround(float _Xx) noexcept { 
#line 176
return ::llroundf(_Xx); 
#line 177
} 
#line 179
[[nodiscard]] inline float log(float _Xx) noexcept { 
#line 180
return ::logf(_Xx); 
#line 181
} 
#line 183
[[nodiscard]] inline float log10(float _Xx) noexcept { 
#line 184
return ::log10f(_Xx); 
#line 185
} 
#line 187
[[nodiscard]] inline float log1p(float _Xx) noexcept { 
#line 188
return ::log1pf(_Xx); 
#line 189
} 
#line 191
[[nodiscard]] inline float log2(float _Xx) noexcept { 
#line 192
return ::log2f(_Xx); 
#line 193
} 
#line 195
[[nodiscard]] inline float logb(float _Xx) noexcept { 
#line 196
return ::logbf(_Xx); 
#line 197
} 
#line 199
[[nodiscard]] inline long lrint(float _Xx) noexcept { 
#line 200
return ::lrintf(_Xx); 
#line 201
} 
#line 203
[[nodiscard]] inline long lround(float _Xx) noexcept { 
#line 204
return ::lroundf(_Xx); 
#line 205
} 
#line 207
inline float modf(float _Xx, float *_Yx) noexcept { 
#line 208
return ::modff(_Xx, _Yx); 
#line 209
} 
#line 211
[[nodiscard]] inline float nearbyint(float _Xx) noexcept { 
#line 212
return ::nearbyintf(_Xx); 
#line 213
} 
#line 215
[[nodiscard]] inline float nextafter(float _Xx, float _Yx) noexcept { 
#line 216
return ::nextafterf(_Xx, _Yx); 
#line 217
} 
#line 219
[[nodiscard]] inline float nexttoward(float _Xx, long double _Yx) noexcept { 
#line 220
return ::nexttowardf(_Xx, _Yx); 
#line 221
} 
#line 223
[[nodiscard]] inline float pow(float _Xx, float _Yx) noexcept { 
#line 224
return ::powf(_Xx, _Yx); 
#line 225
} 
#line 227
[[nodiscard]] inline float remainder(float _Xx, float _Yx) noexcept { 
#line 228
return ::remainderf(_Xx, _Yx); 
#line 229
} 
#line 231
inline float remquo(float _Xx, float _Yx, int *_Zx) noexcept { 
#line 232
return ::remquof(_Xx, _Yx, _Zx); 
#line 233
} 
#line 235
[[nodiscard]] inline float rint(float _Xx) noexcept { 
#line 236
return ::rintf(_Xx); 
#line 237
} 
#line 239
[[nodiscard]] inline float round(float _Xx) noexcept { 
#line 245 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Enterprise\\VC\\Tools\\MSVC\\14.37.32822\\include\\cmath"
return ::roundf(_Xx); 
#line 247 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Enterprise\\VC\\Tools\\MSVC\\14.37.32822\\include\\cmath"
} 
#line 249
[[nodiscard]] inline float scalbln(float _Xx, long _Yx) noexcept { 
#line 250
return ::scalblnf(_Xx, _Yx); 
#line 251
} 
#line 253
[[nodiscard]] inline float scalbn(float _Xx, int _Yx) noexcept { 
#line 254
return ::scalbnf(_Xx, _Yx); 
#line 255
} 
#line 257
[[nodiscard]] inline float sin(float _Xx) noexcept { 
#line 258
return ::sinf(_Xx); 
#line 259
} 
#line 261
[[nodiscard]] inline float sinh(float _Xx) noexcept { 
#line 262
return ::sinhf(_Xx); 
#line 263
} 
#line 265
[[nodiscard]] inline float sqrt(float _Xx) noexcept { 
#line 266
return ::sqrtf(_Xx); 
#line 267
} 
#line 269
[[nodiscard]] inline float tan(float _Xx) noexcept { 
#line 270
return ::tanf(_Xx); 
#line 271
} 
#line 273
[[nodiscard]] inline float tanh(float _Xx) noexcept { 
#line 274
return ::tanhf(_Xx); 
#line 275
} 
#line 277
[[nodiscard]] inline float tgamma(float _Xx) noexcept { 
#line 278
return ::tgammaf(_Xx); 
#line 279
} 
#line 281
[[nodiscard]] inline float trunc(float _Xx) noexcept { 
#line 287 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Enterprise\\VC\\Tools\\MSVC\\14.37.32822\\include\\cmath"
return ::truncf(_Xx); 
#line 289 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Enterprise\\VC\\Tools\\MSVC\\14.37.32822\\include\\cmath"
} 
#line 291
[[nodiscard]] inline long double acos(long double _Xx) noexcept { 
#line 292
return ::acosl(_Xx); 
#line 293
} 
#line 295
[[nodiscard]] inline long double acosh(long double _Xx) noexcept { 
#line 296
return ::acoshl(_Xx); 
#line 297
} 
#line 299
[[nodiscard]] inline long double asin(long double _Xx) noexcept { 
#line 300
return ::asinl(_Xx); 
#line 301
} 
#line 303
[[nodiscard]] inline long double asinh(long double _Xx) noexcept { 
#line 304
return ::asinhl(_Xx); 
#line 305
} 
#line 307
[[nodiscard]] inline long double atan(long double _Xx) noexcept { 
#line 308
return ::atanl(_Xx); 
#line 309
} 
#line 311
[[nodiscard]] inline long double atanh(long double _Xx) noexcept { 
#line 312
return ::atanhl(_Xx); 
#line 313
} 
#line 315
[[nodiscard]] inline long double atan2(long double _Yx, long double _Xx) noexcept 
#line 316
{ 
#line 317
return ::atan2l(_Yx, _Xx); 
#line 318
} 
#line 320
[[nodiscard]] inline long double cbrt(long double _Xx) noexcept { 
#line 321
return ::cbrtl(_Xx); 
#line 322
} 
#line 324
[[nodiscard]] inline long double ceil(long double _Xx) noexcept { 
#line 330 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Enterprise\\VC\\Tools\\MSVC\\14.37.32822\\include\\cmath"
return ::ceill(_Xx); 
#line 332 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Enterprise\\VC\\Tools\\MSVC\\14.37.32822\\include\\cmath"
} 
#line 334
[[nodiscard]] inline long double copysign(long double _Number, long double _Sign) noexcept 
#line 335
{ 
#line 341 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Enterprise\\VC\\Tools\\MSVC\\14.37.32822\\include\\cmath"
return ::copysignl(_Number, _Sign); 
#line 343 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Enterprise\\VC\\Tools\\MSVC\\14.37.32822\\include\\cmath"
} 
#line 345
[[nodiscard]] inline long double cos(long double _Xx) noexcept { 
#line 346
return ::cosl(_Xx); 
#line 347
} 
#line 349
[[nodiscard]] inline long double cosh(long double _Xx) noexcept { 
#line 350
return ::coshl(_Xx); 
#line 351
} 
#line 353
[[nodiscard]] inline long double erf(long double _Xx) noexcept { 
#line 354
return ::erfl(_Xx); 
#line 355
} 
#line 357
[[nodiscard]] inline long double erfc(long double _Xx) noexcept { 
#line 358
return ::erfcl(_Xx); 
#line 359
} 
#line 361
[[nodiscard]] inline long double exp(long double _Xx) noexcept { 
#line 362
return ::expl(_Xx); 
#line 363
} 
#line 365
[[nodiscard]] inline long double exp2(long double _Xx) noexcept { 
#line 366
return ::exp2l(_Xx); 
#line 367
} 
#line 369
[[nodiscard]] inline long double expm1(long double _Xx) noexcept { 
#line 370
return ::expm1l(_Xx); 
#line 371
} 
#line 373
[[nodiscard]] inline long double fabs(long double _Xx) noexcept { 
#line 374
return ::fabsl(_Xx); 
#line 375
} 
#line 377
[[nodiscard]] inline long double fdim(long double _Xx, long double _Yx) noexcept 
#line 378
{ 
#line 379
return ::fdiml(_Xx, _Yx); 
#line 380
} 
#line 382
[[nodiscard]] inline long double floor(long double _Xx) noexcept { 
#line 388 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Enterprise\\VC\\Tools\\MSVC\\14.37.32822\\include\\cmath"
return ::floorl(_Xx); 
#line 390 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Enterprise\\VC\\Tools\\MSVC\\14.37.32822\\include\\cmath"
} 
#line 392
[[nodiscard]] inline long double fma(long double 
#line 393
_Xx, long double _Yx, long double _Zx) noexcept { 
#line 394
return ::fmal(_Xx, _Yx, _Zx); 
#line 395
} 
#line 397
[[nodiscard]] inline long double fmax(long double _Xx, long double _Yx) noexcept 
#line 398
{ 
#line 399
return ::fmaxl(_Xx, _Yx); 
#line 400
} 
#line 402
[[nodiscard]] inline long double fmin(long double _Xx, long double _Yx) noexcept 
#line 403
{ 
#line 404
return ::fminl(_Xx, _Yx); 
#line 405
} 
#line 407
[[nodiscard]] inline long double fmod(long double _Xx, long double _Yx) noexcept 
#line 408
{ 
#line 409
return ::fmodl(_Xx, _Yx); 
#line 410
} 
#line 412
inline long double frexp(long double _Xx, int *_Yx) noexcept { 
#line 413
return ::frexpl(_Xx, _Yx); 
#line 414
} 
#line 416
[[nodiscard]] inline long double hypot(long double _Xx, long double _Yx) noexcept 
#line 417
{ 
#line 418
return ::hypotl(_Xx, _Yx); 
#line 419
} 
#line 421
[[nodiscard]] inline int ilogb(long double _Xx) noexcept { 
#line 422
return ::ilogbl(_Xx); 
#line 423
} 
#line 425
[[nodiscard]] inline long double ldexp(long double _Xx, int _Yx) noexcept { 
#line 426
return ::ldexpl(_Xx, _Yx); 
#line 427
} 
#line 429
[[nodiscard]] inline long double lgamma(long double _Xx) noexcept { 
#line 430
return ::lgammal(_Xx); 
#line 431
} 
#line 433
[[nodiscard]] inline __int64 llrint(long double _Xx) noexcept { 
#line 434
return ::llrintl(_Xx); 
#line 435
} 
#line 437
[[nodiscard]] inline __int64 llround(long double _Xx) noexcept { 
#line 438
return ::llroundl(_Xx); 
#line 439
} 
#line 441
[[nodiscard]] inline long double log(long double _Xx) noexcept { 
#line 442
return ::logl(_Xx); 
#line 443
} 
#line 445
[[nodiscard]] inline long double log10(long double _Xx) noexcept { 
#line 446
return ::log10l(_Xx); 
#line 447
} 
#line 449
[[nodiscard]] inline long double log1p(long double _Xx) noexcept { 
#line 450
return ::log1pl(_Xx); 
#line 451
} 
#line 453
[[nodiscard]] inline long double log2(long double _Xx) noexcept { 
#line 454
return ::log2l(_Xx); 
#line 455
} 
#line 457
[[nodiscard]] inline long double logb(long double _Xx) noexcept { 
#line 458
return ::logbl(_Xx); 
#line 459
} 
#line 461
[[nodiscard]] inline long lrint(long double _Xx) noexcept { 
#line 462
return ::lrintl(_Xx); 
#line 463
} 
#line 465
[[nodiscard]] inline long lround(long double _Xx) noexcept { 
#line 466
return ::lroundl(_Xx); 
#line 467
} 
#line 469
inline long double modf(long double _Xx, long double *_Yx) noexcept { 
#line 470
return ::modfl(_Xx, _Yx); 
#line 471
} 
#line 473
[[nodiscard]] inline long double nearbyint(long double _Xx) noexcept { 
#line 474
return ::nearbyintl(_Xx); 
#line 475
} 
#line 477
[[nodiscard]] inline long double nextafter(long double _Xx, long double _Yx) noexcept 
#line 478
{ 
#line 479
return ::nextafterl(_Xx, _Yx); 
#line 480
} 
#line 482
[[nodiscard]] inline long double nexttoward(long double _Xx, long double _Yx) noexcept 
#line 483
{ 
#line 484
return ::nexttowardl(_Xx, _Yx); 
#line 485
} 
#line 487
[[nodiscard]] inline long double pow(long double _Xx, long double _Yx) noexcept 
#line 488
{ 
#line 489
return ::powl(_Xx, _Yx); 
#line 490
} 
#line 492
[[nodiscard]] inline long double remainder(long double _Xx, long double _Yx) noexcept 
#line 493
{ 
#line 494
return ::remainderl(_Xx, _Yx); 
#line 495
} 
#line 497
inline long double remquo(long double _Xx, long double _Yx, int *_Zx) noexcept { 
#line 498
return ::remquol(_Xx, _Yx, _Zx); 
#line 499
} 
#line 501
[[nodiscard]] inline long double rint(long double _Xx) noexcept { 
#line 502
return ::rintl(_Xx); 
#line 503
} 
#line 505
[[nodiscard]] inline long double round(long double _Xx) noexcept { 
#line 511 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Enterprise\\VC\\Tools\\MSVC\\14.37.32822\\include\\cmath"
return ::roundl(_Xx); 
#line 513 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Enterprise\\VC\\Tools\\MSVC\\14.37.32822\\include\\cmath"
} 
#line 515
[[nodiscard]] inline long double scalbln(long double _Xx, long _Yx) noexcept { 
#line 516
return ::scalblnl(_Xx, _Yx); 
#line 517
} 
#line 519
[[nodiscard]] inline long double scalbn(long double _Xx, int _Yx) noexcept { 
#line 520
return ::scalbnl(_Xx, _Yx); 
#line 521
} 
#line 523
[[nodiscard]] inline long double sin(long double _Xx) noexcept { 
#line 524
return ::sinl(_Xx); 
#line 525
} 
#line 527
[[nodiscard]] inline long double sinh(long double _Xx) noexcept { 
#line 528
return ::sinhl(_Xx); 
#line 529
} 
#line 531
[[nodiscard]] inline long double sqrt(long double _Xx) noexcept { 
#line 532
return ::sqrtl(_Xx); 
#line 533
} 
#line 535
[[nodiscard]] inline long double tan(long double _Xx) noexcept { 
#line 536
return ::tanl(_Xx); 
#line 537
} 
#line 539
[[nodiscard]] inline long double tanh(long double _Xx) noexcept { 
#line 540
return ::tanhl(_Xx); 
#line 541
} 
#line 543
[[nodiscard]] inline long double tgamma(long double _Xx) noexcept { 
#line 544
return ::tgammal(_Xx); 
#line 545
} 
#line 547
[[nodiscard]] inline long double trunc(long double _Xx) noexcept { 
#line 553 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Enterprise\\VC\\Tools\\MSVC\\14.37.32822\\include\\cmath"
return ::truncl(_Xx); 
#line 555 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Enterprise\\VC\\Tools\\MSVC\\14.37.32822\\include\\cmath"
} 
#line 557
namespace std { 
#line 558
template< class _Ty1, class _Ty2> using _Common_float_type_t = conditional_t< is_same_v< _Ty1, long double>  || is_same_v< _Ty2, long double> , long double, conditional_t< is_same_v< _Ty1, float>  && is_same_v< _Ty2, float> , float, double> > ; 
#line 562
}
#line 564
template < class _Ty, :: std :: enable_if_t < :: std :: is_integral_v < _Ty >, int > = 0 >
double frexp ( _Ty _Value, int * const _Exp ) noexcept {
    return :: frexp ( static_cast < double > ( _Value ), _Exp );
}
#line 569
template < class _Ty1, class _Ty2, class _Ty3,
    :: std :: enable_if_t < :: std :: is_arithmetic_v < _Ty1 > && :: std :: is_arithmetic_v < _Ty2 > && :: std :: is_arithmetic_v < _Ty3 >, int > = 0 >
[ [ nodiscard ] ] :: std :: _Common_float_type_t < _Ty1, :: std :: _Common_float_type_t < _Ty2, _Ty3 > > fma (
    _Ty1 _Left, _Ty2 _Middle, _Ty3 _Right ) noexcept {
    using _Common = :: std :: _Common_float_type_t < _Ty1, :: std :: _Common_float_type_t < _Ty2, _Ty3 >>;
    if constexpr ( :: std :: is_same_v < _Common, float > ) {
        return :: fmaf ( static_cast < _Common > ( _Left ), static_cast < _Common > ( _Middle ), static_cast < _Common > ( _Right ) );
    } else if constexpr ( :: std :: is_same_v < _Common, double > ) {
        return :: fma ( static_cast < _Common > ( _Left ), static_cast < _Common > ( _Middle ), static_cast < _Common > ( _Right ) );
    } else {
        return :: fmal ( static_cast < _Common > ( _Left ), static_cast < _Common > ( _Middle ), static_cast < _Common > ( _Right ) );
    }
}
#line 583
template < class _Ty1, class _Ty2, :: std :: enable_if_t < :: std :: is_arithmetic_v < _Ty1 > && :: std :: is_arithmetic_v < _Ty2 >, int > = 0 >
:: std :: _Common_float_type_t < _Ty1, _Ty2 > remquo ( _Ty1 _Left, _Ty2 _Right, int * _Pquo ) noexcept {
    using _Common = :: std :: _Common_float_type_t < _Ty1, _Ty2 >;
    if constexpr ( :: std :: is_same_v < _Common, float > ) {
        return :: remquof ( static_cast < _Common > ( _Left ), static_cast < _Common > ( _Right ), _Pquo );
    } else if constexpr ( :: std :: is_same_v < _Common, double > ) {
        return :: remquo ( static_cast < _Common > ( _Left ), static_cast < _Common > ( _Right ), _Pquo );
    } else {
        return :: remquol ( static_cast < _Common > ( _Left ), static_cast < _Common > ( _Right ), _Pquo );
    }
}
#line 642 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Enterprise\\VC\\Tools\\MSVC\\14.37.32822\\include\\cmath"
template < class _Ty, :: std :: enable_if_t < :: std :: is_integral_v < _Ty >, int > = 0 > [ [ nodiscard ] ] double acos ( _Ty _Left ) noexcept { return :: acos ( static_cast < double > ( _Left ) ); }
#line 643
template < class _Ty, :: std :: enable_if_t < :: std :: is_integral_v < _Ty >, int > = 0 > [ [ nodiscard ] ] double asin ( _Ty _Left ) noexcept { return :: asin ( static_cast < double > ( _Left ) ); }
#line 644
template < class _Ty, :: std :: enable_if_t < :: std :: is_integral_v < _Ty >, int > = 0 > [ [ nodiscard ] ] double atan ( _Ty _Left ) noexcept { return :: atan ( static_cast < double > ( _Left ) ); }
#line 645
template < class _Ty1, class _Ty2, :: std :: enable_if_t < :: std :: is_arithmetic_v < _Ty1 > && :: std :: is_arithmetic_v < _Ty2 >, int > = 0 > [ [ nodiscard ] ] :: std :: _Common_float_type_t < _Ty1, _Ty2 > atan2 ( _Ty1 _Left, _Ty2 _Right ) noexcept { return :: atan2 ( static_cast < double > ( _Left ), static_cast < double > ( _Right ) ); }
#line 646
template < class _Ty, :: std :: enable_if_t < :: std :: is_integral_v < _Ty >, int > = 0 > [ [ nodiscard ] ] double cos ( _Ty _Left ) noexcept { return :: cos ( static_cast < double > ( _Left ) ); }
#line 647
template < class _Ty, :: std :: enable_if_t < :: std :: is_integral_v < _Ty >, int > = 0 > [ [ nodiscard ] ] double sin ( _Ty _Left ) noexcept { return :: sin ( static_cast < double > ( _Left ) ); }
#line 648
template < class _Ty, :: std :: enable_if_t < :: std :: is_integral_v < _Ty >, int > = 0 > [ [ nodiscard ] ] double tan ( _Ty _Left ) noexcept { return :: tan ( static_cast < double > ( _Left ) ); }
#line 649
template < class _Ty, :: std :: enable_if_t < :: std :: is_integral_v < _Ty >, int > = 0 > [ [ nodiscard ] ] double acosh ( _Ty _Left ) noexcept { return :: acosh ( static_cast < double > ( _Left ) ); }
#line 650
template < class _Ty, :: std :: enable_if_t < :: std :: is_integral_v < _Ty >, int > = 0 > [ [ nodiscard ] ] double asinh ( _Ty _Left ) noexcept { return :: asinh ( static_cast < double > ( _Left ) ); }
#line 651
template < class _Ty, :: std :: enable_if_t < :: std :: is_integral_v < _Ty >, int > = 0 > [ [ nodiscard ] ] double atanh ( _Ty _Left ) noexcept { return :: atanh ( static_cast < double > ( _Left ) ); }
#line 652
template < class _Ty, :: std :: enable_if_t < :: std :: is_integral_v < _Ty >, int > = 0 > [ [ nodiscard ] ] double cosh ( _Ty _Left ) noexcept { return :: cosh ( static_cast < double > ( _Left ) ); }
#line 653
template < class _Ty, :: std :: enable_if_t < :: std :: is_integral_v < _Ty >, int > = 0 > [ [ nodiscard ] ] double sinh ( _Ty _Left ) noexcept { return :: sinh ( static_cast < double > ( _Left ) ); }
#line 654
template < class _Ty, :: std :: enable_if_t < :: std :: is_integral_v < _Ty >, int > = 0 > [ [ nodiscard ] ] double tanh ( _Ty _Left ) noexcept { return :: tanh ( static_cast < double > ( _Left ) ); }
#line 655
template < class _Ty, :: std :: enable_if_t < :: std :: is_integral_v < _Ty >, int > = 0 > [ [ nodiscard ] ] double exp ( _Ty _Left ) noexcept { return :: exp ( static_cast < double > ( _Left ) ); }
#line 656
template < class _Ty, :: std :: enable_if_t < :: std :: is_integral_v < _Ty >, int > = 0 > [ [ nodiscard ] ] double exp2 ( _Ty _Left ) noexcept { return :: exp2 ( static_cast < double > ( _Left ) ); }
#line 657
template < class _Ty, :: std :: enable_if_t < :: std :: is_integral_v < _Ty >, int > = 0 > [ [ nodiscard ] ] double expm1 ( _Ty _Left ) noexcept { return :: expm1 ( static_cast < double > ( _Left ) ); }
#line 659
template < class _Ty, :: std :: enable_if_t < :: std :: is_integral_v < _Ty >, int > = 0 > [ [ nodiscard ] ] int ilogb ( _Ty _Left ) noexcept { return :: ilogb ( static_cast < double > ( _Left ) ); }
#line 660
template < class _Ty, :: std :: enable_if_t < :: std :: is_integral_v < _Ty >, int > = 0 > [ [ nodiscard ] ] double ldexp ( _Ty _Left, int _Arg2 ) noexcept { return :: ldexp ( static_cast < double > ( _Left ), _Arg2 ); }
#line 661
template < class _Ty, :: std :: enable_if_t < :: std :: is_integral_v < _Ty >, int > = 0 > [ [ nodiscard ] ] double log ( _Ty _Left ) noexcept { return :: log ( static_cast < double > ( _Left ) ); }
#line 662
template < class _Ty, :: std :: enable_if_t < :: std :: is_integral_v < _Ty >, int > = 0 > [ [ nodiscard ] ] double log10 ( _Ty _Left ) noexcept { return :: log10 ( static_cast < double > ( _Left ) ); }
#line 663
template < class _Ty, :: std :: enable_if_t < :: std :: is_integral_v < _Ty >, int > = 0 > [ [ nodiscard ] ] double log1p ( _Ty _Left ) noexcept { return :: log1p ( static_cast < double > ( _Left ) ); }
#line 664
template < class _Ty, :: std :: enable_if_t < :: std :: is_integral_v < _Ty >, int > = 0 > [ [ nodiscard ] ] double log2 ( _Ty _Left ) noexcept { return :: log2 ( static_cast < double > ( _Left ) ); }
#line 665
template < class _Ty, :: std :: enable_if_t < :: std :: is_integral_v < _Ty >, int > = 0 > [ [ nodiscard ] ] double logb ( _Ty _Left ) noexcept { return :: logb ( static_cast < double > ( _Left ) ); }
#line 667
template < class _Ty, :: std :: enable_if_t < :: std :: is_integral_v < _Ty >, int > = 0 > [ [ nodiscard ] ] double scalbn ( _Ty _Left, int _Arg2 ) noexcept { return :: scalbn ( static_cast < double > ( _Left ), _Arg2 ); }
#line 668
template < class _Ty, :: std :: enable_if_t < :: std :: is_integral_v < _Ty >, int > = 0 > [ [ nodiscard ] ] double scalbln ( _Ty _Left, long _Arg2 ) noexcept { return :: scalbln ( static_cast < double > ( _Left ), _Arg2 ); }
#line 669
template < class _Ty, :: std :: enable_if_t < :: std :: is_integral_v < _Ty >, int > = 0 > [ [ nodiscard ] ] double cbrt ( _Ty _Left ) noexcept { return :: cbrt ( static_cast < double > ( _Left ) ); }
#line 671
template < class _Ty, :: std :: enable_if_t < :: std :: is_integral_v < _Ty >, int > = 0 > [ [ nodiscard ] ] double fabs ( _Ty _Left ) noexcept { return :: fabs ( static_cast < double > ( _Left ) ); }
#line 672
template < class _Ty1, class _Ty2, :: std :: enable_if_t < :: std :: is_arithmetic_v < _Ty1 > && :: std :: is_arithmetic_v < _Ty2 >, int > = 0 > [ [ nodiscard ] ] :: std :: _Common_float_type_t < _Ty1, _Ty2 > hypot ( _Ty1 _Left, _Ty2 _Right ) noexcept { return :: hypot ( static_cast < double > ( _Left ), static_cast < double > ( _Right ) ); }
#line 674
template < class _Ty1, class _Ty2, :: std :: enable_if_t < :: std :: is_arithmetic_v < _Ty1 > && :: std :: is_arithmetic_v < _Ty2 >, int > = 0 > [ [ nodiscard ] ] :: std :: _Common_float_type_t < _Ty1, _Ty2 > pow ( _Ty1 _Left, _Ty2 _Right ) noexcept { return :: pow ( static_cast < double > ( _Left ), static_cast < double > ( _Right ) ); }
#line 675
template < class _Ty, :: std :: enable_if_t < :: std :: is_integral_v < _Ty >, int > = 0 > [ [ nodiscard ] ] double sqrt ( _Ty _Left ) noexcept { return :: sqrt ( static_cast < double > ( _Left ) ); }
#line 676
template < class _Ty, :: std :: enable_if_t < :: std :: is_integral_v < _Ty >, int > = 0 > [ [ nodiscard ] ] double erf ( _Ty _Left ) noexcept { return :: erf ( static_cast < double > ( _Left ) ); }
#line 677
template < class _Ty, :: std :: enable_if_t < :: std :: is_integral_v < _Ty >, int > = 0 > [ [ nodiscard ] ] double erfc ( _Ty _Left ) noexcept { return :: erfc ( static_cast < double > ( _Left ) ); }
#line 678
template < class _Ty, :: std :: enable_if_t < :: std :: is_integral_v < _Ty >, int > = 0 > [ [ nodiscard ] ] double lgamma ( _Ty _Left ) noexcept { return :: lgamma ( static_cast < double > ( _Left ) ); }
#line 679
template < class _Ty, :: std :: enable_if_t < :: std :: is_integral_v < _Ty >, int > = 0 > [ [ nodiscard ] ] double tgamma ( _Ty _Left ) noexcept { return :: tgamma ( static_cast < double > ( _Left ) ); }
#line 680
template < class _Ty, :: std :: enable_if_t < :: std :: is_integral_v < _Ty >, int > = 0 > [ [ nodiscard ] ] double ceil ( _Ty _Left ) noexcept { return :: ceil ( static_cast < double > ( _Left ) ); }
#line 681
template < class _Ty, :: std :: enable_if_t < :: std :: is_integral_v < _Ty >, int > = 0 > [ [ nodiscard ] ] double floor ( _Ty _Left ) noexcept { return :: floor ( static_cast < double > ( _Left ) ); }
#line 682
template < class _Ty, :: std :: enable_if_t < :: std :: is_integral_v < _Ty >, int > = 0 > [ [ nodiscard ] ] double nearbyint ( _Ty _Left ) noexcept { return :: nearbyint ( static_cast < double > ( _Left ) ); }
#line 683
template < class _Ty, :: std :: enable_if_t < :: std :: is_integral_v < _Ty >, int > = 0 > [ [ nodiscard ] ] double rint ( _Ty _Left ) noexcept { return :: rint ( static_cast < double > ( _Left ) ); }
#line 684
template < class _Ty, :: std :: enable_if_t < :: std :: is_integral_v < _Ty >, int > = 0 > [ [ nodiscard ] ] long lrint ( _Ty _Left ) noexcept { return :: lrint ( static_cast < double > ( _Left ) ); }
#line 685
template < class _Ty, :: std :: enable_if_t < :: std :: is_integral_v < _Ty >, int > = 0 > [ [ nodiscard ] ] long long llrint ( _Ty _Left ) noexcept { return :: llrint ( static_cast < double > ( _Left ) ); }
#line 686
template < class _Ty, :: std :: enable_if_t < :: std :: is_integral_v < _Ty >, int > = 0 > [ [ nodiscard ] ] double round ( _Ty _Left ) noexcept { return :: round ( static_cast < double > ( _Left ) ); }
#line 687
template < class _Ty, :: std :: enable_if_t < :: std :: is_integral_v < _Ty >, int > = 0 > [ [ nodiscard ] ] long lround ( _Ty _Left ) noexcept { return :: lround ( static_cast < double > ( _Left ) ); }
#line 688
template < class _Ty, :: std :: enable_if_t < :: std :: is_integral_v < _Ty >, int > = 0 > [ [ nodiscard ] ] long long llround ( _Ty _Left ) noexcept { return :: llround ( static_cast < double > ( _Left ) ); }
#line 689
template < class _Ty, :: std :: enable_if_t < :: std :: is_integral_v < _Ty >, int > = 0 > [ [ nodiscard ] ] double trunc ( _Ty _Left ) noexcept { return :: trunc ( static_cast < double > ( _Left ) ); }
#line 690
template < class _Ty1, class _Ty2, :: std :: enable_if_t < :: std :: is_arithmetic_v < _Ty1 > && :: std :: is_arithmetic_v < _Ty2 >, int > = 0 > [ [ nodiscard ] ] :: std :: _Common_float_type_t < _Ty1, _Ty2 > fmod ( _Ty1 _Left, _Ty2 _Right ) noexcept { return :: fmod ( static_cast < double > ( _Left ), static_cast < double > ( _Right ) ); }
#line 691
template < class _Ty1, class _Ty2, :: std :: enable_if_t < :: std :: is_arithmetic_v < _Ty1 > && :: std :: is_arithmetic_v < _Ty2 >, int > = 0 > [ [ nodiscard ] ] :: std :: _Common_float_type_t < _Ty1, _Ty2 > remainder ( _Ty1 _Left, _Ty2 _Right ) noexcept { return :: remainder ( static_cast < double > ( _Left ), static_cast < double > ( _Right ) ); }
#line 693
template < class _Ty1, class _Ty2, :: std :: enable_if_t < :: std :: is_arithmetic_v < _Ty1 > && :: std :: is_arithmetic_v < _Ty2 >, int > = 0 > [ [ nodiscard ] ] :: std :: _Common_float_type_t < _Ty1, _Ty2 > copysign ( _Ty1 _Left, _Ty2 _Right ) noexcept { return :: copysign ( static_cast < double > ( _Left ), static_cast < double > ( _Right ) ); }
#line 695
template < class _Ty1, class _Ty2, :: std :: enable_if_t < :: std :: is_arithmetic_v < _Ty1 > && :: std :: is_arithmetic_v < _Ty2 >, int > = 0 > [ [ nodiscard ] ] :: std :: _Common_float_type_t < _Ty1, _Ty2 > nextafter ( _Ty1 _Left, _Ty2 _Right ) noexcept { return :: nextafter ( static_cast < double > ( _Left ), static_cast < double > ( _Right ) ); }
#line 696
template < class _Ty, :: std :: enable_if_t < :: std :: is_integral_v < _Ty >, int > = 0 > [ [ nodiscard ] ] double nexttoward ( _Ty _Left, long double _Arg2 ) noexcept { return :: nexttoward ( static_cast < double > ( _Left ), _Arg2 ); }
#line 697
template < class _Ty1, class _Ty2, :: std :: enable_if_t < :: std :: is_arithmetic_v < _Ty1 > && :: std :: is_arithmetic_v < _Ty2 >, int > = 0 > [ [ nodiscard ] ] :: std :: _Common_float_type_t < _Ty1, _Ty2 > fdim ( _Ty1 _Left, _Ty2 _Right ) noexcept { return :: fdim ( static_cast < double > ( _Left ), static_cast < double > ( _Right ) ); }
#line 698
template < class _Ty1, class _Ty2, :: std :: enable_if_t < :: std :: is_arithmetic_v < _Ty1 > && :: std :: is_arithmetic_v < _Ty2 >, int > = 0 > [ [ nodiscard ] ] :: std :: _Common_float_type_t < _Ty1, _Ty2 > fmax ( _Ty1 _Left, _Ty2 _Right ) noexcept { return :: fmax ( static_cast < double > ( _Left ), static_cast < double > ( _Right ) ); }
#line 699
template < class _Ty1, class _Ty2, :: std :: enable_if_t < :: std :: is_arithmetic_v < _Ty1 > && :: std :: is_arithmetic_v < _Ty2 >, int > = 0 > [ [ nodiscard ] ] :: std :: _Common_float_type_t < _Ty1, _Ty2 > fmin ( _Ty1 _Left, _Ty2 _Right ) noexcept { return :: fmin ( static_cast < double > ( _Left ), static_cast < double > ( _Right ) ); }
#line 714
namespace std { 
#line 715
using ::abs;
#line 716
using ::acos;
#line 717
using ::asin;
#line 718
using ::atan;
#line 719
using ::atan2;
#line 720
using ::ceil;
#line 721
using ::cos;
#line 722
using ::cosh;
#line 723
using ::exp;
#line 724
using ::fabs;
#line 725
using ::floor;
#line 726
using ::fmod;
#line 727
using ::frexp;
#line 728
using ::ldexp;
#line 729
using ::log;
#line 730
using ::log10;
#line 731
using ::modf;
#line 732
using ::pow;
#line 733
using ::sin;
#line 734
using ::sinh;
#line 735
using ::sqrt;
#line 736
using ::tan;
#line 737
using ::tanh;
#line 739
using ::acosf;
#line 740
using ::asinf;
#line 741
using ::atanf;
#line 742
using ::atan2f;
#line 743
using ::ceilf;
#line 744
using ::cosf;
#line 745
using ::coshf;
#line 746
using ::expf;
#line 747
using ::fabsf;
#line 748
using ::floorf;
#line 749
using ::fmodf;
#line 750
using ::frexpf;
#line 751
using ::ldexpf;
#line 752
using ::logf;
#line 753
using ::log10f;
#line 754
using ::modff;
#line 755
using ::powf;
#line 756
using ::sinf;
#line 757
using ::sinhf;
#line 758
using ::sqrtf;
#line 759
using ::tanf;
#line 760
using ::tanhf;
#line 762
using ::acosl;
#line 763
using ::asinl;
#line 764
using ::atanl;
#line 765
using ::atan2l;
#line 766
using ::ceill;
#line 767
using ::cosl;
#line 768
using ::coshl;
#line 769
using ::expl;
#line 770
using ::fabsl;
#line 771
using ::floorl;
#line 772
using ::fmodl;
#line 773
using ::frexpl;
#line 774
using ::ldexpl;
#line 775
using ::logl;
#line 776
using ::log10l;
#line 777
using ::modfl;
#line 778
using ::powl;
#line 779
using ::sinl;
#line 780
using ::sinhl;
#line 781
using ::sqrtl;
#line 782
using ::tanl;
#line 783
using ::tanhl;
#line 785
using ::float_t;
#line 786
using ::double_t;
#line 788
using ::acosh;
#line 789
using ::asinh;
#line 790
using ::atanh;
#line 791
using ::cbrt;
#line 792
using ::erf;
#line 793
using ::erfc;
#line 794
using ::expm1;
#line 795
using ::exp2;
#line 797
inline namespace _Binary_hypot { 
#line 798
using ::hypot;
#line 799
}
#line 801
using ::ilogb;
#line 802
using ::lgamma;
#line 803
using ::log1p;
#line 804
using ::log2;
#line 805
using ::logb;
#line 806
using ::llrint;
#line 807
using ::lrint;
#line 808
using ::nearbyint;
#line 809
using ::rint;
#line 810
using ::llround;
#line 811
using ::lround;
#line 812
using ::fdim;
#line 813
using ::fma;
#line 814
using ::fmax;
#line 815
using ::fmin;
#line 816
using ::round;
#line 817
using ::trunc;
#line 818
using ::remainder;
#line 819
using ::remquo;
#line 820
using ::copysign;
#line 821
using ::nan;
#line 822
using ::nextafter;
#line 823
using ::scalbn;
#line 824
using ::scalbln;
#line 825
using ::nexttoward;
#line 826
using ::tgamma;
#line 828
using ::acoshf;
#line 829
using ::asinhf;
#line 830
using ::atanhf;
#line 831
using ::cbrtf;
#line 832
using ::erff;
#line 833
using ::erfcf;
#line 834
using ::expm1f;
#line 835
using ::exp2f;
#line 836
using ::hypotf;
#line 837
using ::ilogbf;
#line 838
using ::lgammaf;
#line 839
using ::log1pf;
#line 840
using ::log2f;
#line 841
using ::logbf;
#line 842
using ::llrintf;
#line 843
using ::lrintf;
#line 844
using ::nearbyintf;
#line 845
using ::rintf;
#line 846
using ::llroundf;
#line 847
using ::lroundf;
#line 848
using ::fdimf;
#line 849
using ::fmaf;
#line 850
using ::fmaxf;
#line 851
using ::fminf;
#line 852
using ::roundf;
#line 853
using ::truncf;
#line 854
using ::remainderf;
#line 855
using ::remquof;
#line 856
using ::copysignf;
#line 857
using ::nanf;
#line 858
using ::nextafterf;
#line 859
using ::scalbnf;
#line 860
using ::scalblnf;
#line 861
using ::nexttowardf;
#line 862
using ::tgammaf;
#line 864
using ::acoshl;
#line 865
using ::asinhl;
#line 866
using ::atanhl;
#line 867
using ::cbrtl;
#line 868
using ::erfl;
#line 869
using ::erfcl;
#line 870
using ::expm1l;
#line 871
using ::exp2l;
#line 872
using ::hypotl;
#line 873
using ::ilogbl;
#line 874
using ::lgammal;
#line 875
using ::log1pl;
#line 876
using ::log2l;
#line 877
using ::logbl;
#line 878
using ::llrintl;
#line 879
using ::lrintl;
#line 880
using ::nearbyintl;
#line 881
using ::rintl;
#line 882
using ::llroundl;
#line 883
using ::lroundl;
#line 884
using ::fdiml;
#line 885
using ::fmal;
#line 886
using ::fmaxl;
#line 887
using ::fminl;
#line 888
using ::roundl;
#line 889
using ::truncl;
#line 890
using ::remainderl;
#line 891
using ::remquol;
#line 892
using ::copysignl;
#line 893
using ::nanl;
#line 894
using ::nextafterl;
#line 895
using ::scalbnl;
#line 896
using ::scalblnl;
#line 897
using ::nexttowardl;
#line 898
using ::tgammal;
#line 900
using ::fpclassify;
#line 901
using ::signbit;
#line 902
using ::isfinite;
#line 903
using ::isinf;
#line 904
using ::isnan;
#line 905
using ::isnormal;
#line 906
using ::isgreater;
#line 907
using ::isgreaterequal;
#line 908
using ::isless;
#line 909
using ::islessequal;
#line 910
using ::islessgreater;
#line 911
using ::isunordered;
#line 912
}
#line 1487 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Enterprise\\VC\\Tools\\MSVC\\14.37.32822\\include\\cmath"
#pragma warning(pop)
#pragma pack ( pop )
#line 10832 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.h"
extern "C" double __cdecl _hypot(double x, double y); 
#line 10833
extern "C" float __cdecl _hypotf(float x, float y); 
#line 10843 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.h"
extern inline bool signbit(long double) throw(); 
#line 10844
extern "C" int _ldsign(long double); 
#line 10887 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.h"
extern inline bool signbit(double) throw(); 
#line 10888
extern "C" int _dsign(double); 
#line 10932 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.h"
extern inline bool signbit(float) throw(); 
#line 10933
extern "C" int _fdsign(float); 
#line 10941 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.h"
static __inline bool isinf(long double a); 
#line 10976 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.h"
static __inline bool isinf(double a); 
#line 11014 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.h"
static __inline bool isinf(float a); 
#line 11021 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.h"
static __inline bool isnan(long double a); 
#line 11054 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.h"
static __inline bool isnan(double a); 
#line 11090 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.h"
static __inline bool isnan(float a); 
#line 11097 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.h"
static __inline bool isfinite(long double a); 
#line 11134 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.h"
static __inline bool isfinite(double a); 
#line 11170 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.h"
static __inline bool isfinite(float a); 
#line 11178 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.h"
template< class T> extern T _Pow_int(T, int) throw(); 
#line 11179
extern inline __int64 abs(__int64) throw(); 
#line 11260 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.h"
extern inline long __cdecl abs(long) throw(); 
#line 11264 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.h"
extern inline float __cdecl abs(float) throw(); 
#line 11265
extern inline double __cdecl abs(double) throw(); 
#line 11266
extern inline float __cdecl fabs(float) throw(); 
#line 11267
extern inline float __cdecl ceil(float) throw(); 
#line 11268
extern inline float __cdecl floor(float) throw(); 
#line 11269
extern inline float __cdecl sqrt(float) throw(); 
#line 11270
extern inline float __cdecl pow(float, float) throw(); 
#line 11299 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.h"
template< class _Ty1, class _Ty2, std::enable_if_t< std::is_arithmetic_v< _Ty1>  && std::is_arithmetic_v< _Ty2> , int>  > [[nodiscard]] std::_Common_float_type_t< _Ty1, _Ty2>  __cdecl pow(_Ty1 _Left, _Ty2 _Right) noexcept; 
#line 11306 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.h"
extern inline float __cdecl log(float) throw(); 
#line 11307
extern inline float __cdecl log10(float) throw(); 
#line 11308
extern inline float __cdecl fmod(float, float) throw(); 
#line 11309
extern inline float __cdecl modf(float, float *) throw(); 
#line 11310
extern inline float __cdecl exp(float) throw(); 
#line 11311
extern inline float __cdecl frexp(float, int *) throw(); 
#line 11312
extern inline float __cdecl ldexp(float, int) throw(); 
#line 11313
extern inline float __cdecl asin(float) throw(); 
#line 11314
extern inline float __cdecl sin(float) throw(); 
#line 11315
extern inline float __cdecl sinh(float) throw(); 
#line 11316
extern inline float __cdecl acos(float) throw(); 
#line 11317
extern inline float __cdecl cos(float) throw(); 
#line 11318
extern inline float __cdecl cosh(float) throw(); 
#line 11319
extern inline float __cdecl atan(float) throw(); 
#line 11320
extern inline float __cdecl atan2(float, float) throw(); 
#line 11321
extern inline float __cdecl tan(float) throw(); 
#line 11322
extern inline float __cdecl tanh(float) throw(); 
#line 11545 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.h"
extern inline float __cdecl logb(float) throw(); 
#line 11546
extern inline int __cdecl ilogb(float) throw(); 
#line 11547
extern float __cdecl scalbn(float, float) throw(); 
#line 11548
extern inline float __cdecl scalbln(float, long) throw(); 
#line 11549
extern inline float __cdecl exp2(float) throw(); 
#line 11550
extern inline float __cdecl expm1(float) throw(); 
#line 11551
extern inline float __cdecl log2(float) throw(); 
#line 11552
extern inline float __cdecl log1p(float) throw(); 
#line 11553
extern inline float __cdecl acosh(float) throw(); 
#line 11554
extern inline float __cdecl asinh(float) throw(); 
#line 11555
extern inline float __cdecl atanh(float) throw(); 
#line 11556
extern inline float __cdecl hypot(float, float) throw(); 
#line 11557
extern inline float __cdecl cbrt(float) throw(); 
#line 11558
extern inline float __cdecl erf(float) throw(); 
#line 11559
extern inline float __cdecl erfc(float) throw(); 
#line 11560
extern inline float __cdecl lgamma(float) throw(); 
#line 11561
extern inline float __cdecl tgamma(float) throw(); 
#line 11562
extern inline float __cdecl copysign(float, float) throw(); 
#line 11563
extern inline float __cdecl nextafter(float, float) throw(); 
#line 11564
extern inline float __cdecl remainder(float, float) throw(); 
#line 11565
extern inline float __cdecl remquo(float, float, int *) throw(); 
#line 11566
extern inline float __cdecl round(float) throw(); 
#line 11567
extern inline long __cdecl lround(float) throw(); 
#line 11568
extern inline __int64 __cdecl llround(float) throw(); 
#line 11569
extern inline float __cdecl trunc(float) throw(); 
#line 11570
extern inline float __cdecl rint(float) throw(); 
#line 11571
extern inline long __cdecl lrint(float) throw(); 
#line 11572
extern inline __int64 __cdecl llrint(float) throw(); 
#line 11573
extern inline float __cdecl nearbyint(float) throw(); 
#line 11574
extern inline float __cdecl fdim(float, float) throw(); 
#line 11575
extern inline float __cdecl fma(float, float, float) throw(); 
#line 11576
extern inline float __cdecl fmax(float, float) throw(); 
#line 11577
extern inline float __cdecl fmin(float, float) throw(); 
#line 11580 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.h"
static inline float exp10(const float a); 
#line 11582
static inline float rsqrt(const float a); 
#line 11584
static inline float rcbrt(const float a); 
#line 11586
static inline float sinpi(const float a); 
#line 11588
static inline float cospi(const float a); 
#line 11590
static inline void sincospi(const float a, float *const sptr, float *const cptr); 
#line 11592
static inline void sincos(const float a, float *const sptr, float *const cptr); 
#line 11594
static inline float j0(const float a); 
#line 11596
static inline float j1(const float a); 
#line 11598
static inline float jn(const int n, const float a); 
#line 11600
static inline float y0(const float a); 
#line 11602
static inline float y1(const float a); 
#line 11604
static inline float yn(const int n, const float a); 
#line 11606
static inline float cyl_bessel_i0(const float a); 
#line 11608
static inline float cyl_bessel_i1(const float a); 
#line 11610
static inline float erfinv(const float a); 
#line 11612
static inline float erfcinv(const float a); 
#line 11614
static inline float normcdfinv(const float a); 
#line 11616
static inline float normcdf(const float a); 
#line 11618
static inline float erfcx(const float a); 
#line 11620
static inline double copysign(const double a, const float b); 
#line 11622
static inline double copysign(const float a, const double b); 
#line 11630
static inline unsigned min(const unsigned a, const unsigned b); 
#line 11638
static inline unsigned min(const int a, const unsigned b); 
#line 11646
static inline unsigned min(const unsigned a, const int b); 
#line 11654
static inline long min(const long a, const long b); 
#line 11662
static inline unsigned long min(const unsigned long a, const unsigned long b); 
#line 11670
static inline unsigned long min(const long a, const unsigned long b); 
#line 11678
static inline unsigned long min(const unsigned long a, const long b); 
#line 11686
static inline __int64 min(const __int64 a, const __int64 b); 
#line 11694
static inline unsigned __int64 min(const unsigned __int64 a, const unsigned __int64 b); 
#line 11702
static inline unsigned __int64 min(const __int64 a, const unsigned __int64 b); 
#line 11710
static inline unsigned __int64 min(const unsigned __int64 a, const __int64 b); 
#line 11721
static inline float min(const float a, const float b); 
#line 11732
static inline double min(const double a, const double b); 
#line 11742
static inline double min(const float a, const double b); 
#line 11752
static inline double min(const double a, const float b); 
#line 11760
static inline unsigned max(const unsigned a, const unsigned b); 
#line 11768
static inline unsigned max(const int a, const unsigned b); 
#line 11776
static inline unsigned max(const unsigned a, const int b); 
#line 11784
static inline long max(const long a, const long b); 
#line 11792
static inline unsigned long max(const unsigned long a, const unsigned long b); 
#line 11800
static inline unsigned long max(const long a, const unsigned long b); 
#line 11808
static inline unsigned long max(const unsigned long a, const long b); 
#line 11816
static inline __int64 max(const __int64 a, const __int64 b); 
#line 11824
static inline unsigned __int64 max(const unsigned __int64 a, const unsigned __int64 b); 
#line 11832
static inline unsigned __int64 max(const __int64 a, const unsigned __int64 b); 
#line 11840
static inline unsigned __int64 max(const unsigned __int64 a, const __int64 b); 
#line 11851
static inline float max(const float a, const float b); 
#line 11862
static inline double max(const double a, const double b); 
#line 11872
static inline double max(const float a, const double b); 
#line 11882
static inline double max(const double a, const float b); 
#line 11893
extern "C" {
#line 11894
inline void *__nv_aligned_device_malloc(size_t size, size_t align) {int volatile ___ = 1;(void)size;(void)align;::exit(___);}
#if 0
#line 11895
{ 
#line 11896
void *__nv_aligned_device_malloc_impl(size_t, size_t); 
#line 11897
return __nv_aligned_device_malloc_impl(size, align); 
#line 11898
} 
#endif
#line 11899 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.h"
}
#line 433 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.hpp"
static __inline bool isinf(const long double a) 
#line 434
{ 
#line 438 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.hpp"
return isinf< long double> (a); 
#line 440 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.hpp"
} 
#line 449 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.hpp"
static __inline bool isinf(const double a) 
#line 450
{ 
#line 454 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.hpp"
return isinf< double> (a); 
#line 456 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.hpp"
} 
#line 465 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.hpp"
static __inline bool isinf(const float a) 
#line 466
{ 
#line 470 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.hpp"
return isinf< float> (a); 
#line 472 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.hpp"
} 
#line 481 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.hpp"
static __inline bool isnan(const long double a) 
#line 482
{ 
#line 486 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.hpp"
return isnan< long double> (a); 
#line 488 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.hpp"
} 
#line 497 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.hpp"
static __inline bool isnan(const double a) 
#line 498
{ 
#line 502 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.hpp"
return isnan< double> (a); 
#line 504 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.hpp"
} 
#line 513 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.hpp"
static __inline bool isnan(const float a) 
#line 514
{ 
#line 518 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.hpp"
return isnan< float> (a); 
#line 520 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.hpp"
} 
#line 529 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.hpp"
static __inline bool isfinite(const long double a) 
#line 530
{ 
#line 534 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.hpp"
return isfinite< long double> (a); 
#line 536 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.hpp"
} 
#line 545 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.hpp"
static __inline bool isfinite(const double a) 
#line 546
{ 
#line 550 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.hpp"
return isfinite< double> (a); 
#line 552 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.hpp"
} 
#line 561 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.hpp"
static __inline bool isfinite(const float a) 
#line 562
{ 
#line 566 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.hpp"
return isfinite< float> (a); 
#line 568 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.hpp"
} 
#line 758 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.hpp"
static inline float exp10(const float a) 
#line 759
{ 
#line 760
return exp10f(a); 
#line 761
} 
#line 763
static inline float rsqrt(const float a) 
#line 764
{ 
#line 765
return rsqrtf(a); 
#line 766
} 
#line 768
static inline float rcbrt(const float a) 
#line 769
{ 
#line 770
return rcbrtf(a); 
#line 771
} 
#line 773
static inline float sinpi(const float a) 
#line 774
{ 
#line 775
return sinpif(a); 
#line 776
} 
#line 778
static inline float cospi(const float a) 
#line 779
{ 
#line 780
return cospif(a); 
#line 781
} 
#line 783
static inline void sincospi(const float a, float *const sptr, float *const cptr) 
#line 784
{ 
#line 785
sincospif(a, sptr, cptr); 
#line 786
} 
#line 788
static inline void sincos(const float a, float *const sptr, float *const cptr) 
#line 789
{ 
#line 790
sincosf(a, sptr, cptr); 
#line 791
} 
#line 793
static inline float j0(const float a) 
#line 794
{ 
#line 795
return j0f(a); 
#line 796
} 
#line 798
static inline float j1(const float a) 
#line 799
{ 
#line 800
return j1f(a); 
#line 801
} 
#line 803
static inline float jn(const int n, const float a) 
#line 804
{ 
#line 805
return jnf(n, a); 
#line 806
} 
#line 808
static inline float y0(const float a) 
#line 809
{ 
#line 810
return y0f(a); 
#line 811
} 
#line 813
static inline float y1(const float a) 
#line 814
{ 
#line 815
return y1f(a); 
#line 816
} 
#line 818
static inline float yn(const int n, const float a) 
#line 819
{ 
#line 820
return ynf(n, a); 
#line 821
} 
#line 823
static inline float cyl_bessel_i0(const float a) {int volatile ___ = 1;(void)a;::exit(___);}
#if 0
#line 824
{ 
#line 825
return cyl_bessel_i0f(a); 
#line 826
} 
#endif
#line 828 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.hpp"
static inline float cyl_bessel_i1(const float a) {int volatile ___ = 1;(void)a;::exit(___);}
#if 0
#line 829
{ 
#line 830
return cyl_bessel_i1f(a); 
#line 831
} 
#endif
#line 833 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.hpp"
static inline float erfinv(const float a) 
#line 834
{ 
#line 835
return erfinvf(a); 
#line 836
} 
#line 838
static inline float erfcinv(const float a) 
#line 839
{ 
#line 840
return erfcinvf(a); 
#line 841
} 
#line 843
static inline float normcdfinv(const float a) 
#line 844
{ 
#line 845
return normcdfinvf(a); 
#line 846
} 
#line 848
static inline float normcdf(const float a) 
#line 849
{ 
#line 850
return normcdff(a); 
#line 851
} 
#line 853
static inline float erfcx(const float a) 
#line 854
{ 
#line 855
return erfcxf(a); 
#line 856
} 
#line 858
static inline double copysign(const double a, const float b) 
#line 859
{ 
#line 860
return copysign(a, static_cast< double>(b)); 
#line 861
} 
#line 863
static inline double copysign(const float a, const double b) 
#line 864
{ 
#line 865
return copysign(static_cast< double>(a), b); 
#line 866
} 
#line 868
static inline unsigned min(const unsigned a, const unsigned b) 
#line 869
{ 
#line 870
return umin(a, b); 
#line 871
} 
#line 873
static inline unsigned min(const int a, const unsigned b) 
#line 874
{ 
#line 875
return umin(static_cast< unsigned>(a), b); 
#line 876
} 
#line 878
static inline unsigned min(const unsigned a, const int b) 
#line 879
{ 
#line 880
return umin(a, static_cast< unsigned>(b)); 
#line 881
} 
#line 883
static inline long min(const long a, const long b) 
#line 884
{ 
#line 885
long retval; 
#line 888
#pragma warning (disable: 4127)
#line 891 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.hpp"
if (sizeof(long) == sizeof(int)) { 
#line 893
#pragma warning (default: 4127)
#line 895 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.hpp"
retval = (static_cast< long>(min(static_cast< int>(a), static_cast< int>(b)))); 
#line 896
} else { 
#line 897
retval = (static_cast< long>(llmin(static_cast< __int64>(a), static_cast< __int64>(b)))); 
#line 898
}  
#line 899
return retval; 
#line 900
} 
#line 902
static inline unsigned long min(const unsigned long a, const unsigned long b) 
#line 903
{ 
#line 904
unsigned long retval; 
#line 906
#pragma warning (disable: 4127)
#line 908 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.hpp"
if (sizeof(unsigned long) == sizeof(unsigned)) { 
#line 910
#pragma warning (default: 4127)
#line 912 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.hpp"
retval = (static_cast< unsigned long>(umin(static_cast< unsigned>(a), static_cast< unsigned>(b)))); 
#line 913
} else { 
#line 914
retval = (static_cast< unsigned long>(ullmin(static_cast< unsigned __int64>(a), static_cast< unsigned __int64>(b)))); 
#line 915
}  
#line 916
return retval; 
#line 917
} 
#line 919
static inline unsigned long min(const long a, const unsigned long b) 
#line 920
{ 
#line 921
unsigned long retval; 
#line 923
#pragma warning (disable: 4127)
#line 925 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.hpp"
if (sizeof(unsigned long) == sizeof(unsigned)) { 
#line 927
#pragma warning (default: 4127)
#line 929 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.hpp"
retval = (static_cast< unsigned long>(umin(static_cast< unsigned>(a), static_cast< unsigned>(b)))); 
#line 930
} else { 
#line 931
retval = (static_cast< unsigned long>(ullmin(static_cast< unsigned __int64>(a), static_cast< unsigned __int64>(b)))); 
#line 932
}  
#line 933
return retval; 
#line 934
} 
#line 936
static inline unsigned long min(const unsigned long a, const long b) 
#line 937
{ 
#line 938
unsigned long retval; 
#line 940
#pragma warning (disable: 4127)
#line 942 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.hpp"
if (sizeof(unsigned long) == sizeof(unsigned)) { 
#line 944
#pragma warning (default: 4127)
#line 946 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.hpp"
retval = (static_cast< unsigned long>(umin(static_cast< unsigned>(a), static_cast< unsigned>(b)))); 
#line 947
} else { 
#line 948
retval = (static_cast< unsigned long>(ullmin(static_cast< unsigned __int64>(a), static_cast< unsigned __int64>(b)))); 
#line 949
}  
#line 950
return retval; 
#line 951
} 
#line 953
static inline __int64 min(const __int64 a, const __int64 b) 
#line 954
{ 
#line 955
return llmin(a, b); 
#line 956
} 
#line 958
static inline unsigned __int64 min(const unsigned __int64 a, const unsigned __int64 b) 
#line 959
{ 
#line 960
return ullmin(a, b); 
#line 961
} 
#line 963
static inline unsigned __int64 min(const __int64 a, const unsigned __int64 b) 
#line 964
{ 
#line 965
return ullmin(static_cast< unsigned __int64>(a), b); 
#line 966
} 
#line 968
static inline unsigned __int64 min(const unsigned __int64 a, const __int64 b) 
#line 969
{ 
#line 970
return ullmin(a, static_cast< unsigned __int64>(b)); 
#line 971
} 
#line 973
static inline float min(const float a, const float b) 
#line 974
{ 
#line 975
return fminf(a, b); 
#line 976
} 
#line 978
static inline double min(const double a, const double b) 
#line 979
{ 
#line 980
return fmin(a, b); 
#line 981
} 
#line 983
static inline double min(const float a, const double b) 
#line 984
{ 
#line 985
return fmin(static_cast< double>(a), b); 
#line 986
} 
#line 988
static inline double min(const double a, const float b) 
#line 989
{ 
#line 990
return fmin(a, static_cast< double>(b)); 
#line 991
} 
#line 993
static inline unsigned max(const unsigned a, const unsigned b) 
#line 994
{ 
#line 995
return umax(a, b); 
#line 996
} 
#line 998
static inline unsigned max(const int a, const unsigned b) 
#line 999
{ 
#line 1000
return umax(static_cast< unsigned>(a), b); 
#line 1001
} 
#line 1003
static inline unsigned max(const unsigned a, const int b) 
#line 1004
{ 
#line 1005
return umax(a, static_cast< unsigned>(b)); 
#line 1006
} 
#line 1008
static inline long max(const long a, const long b) 
#line 1009
{ 
#line 1010
long retval; 
#line 1013
#pragma warning (disable: 4127)
#line 1015 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.hpp"
if (sizeof(long) == sizeof(int)) { 
#line 1017
#pragma warning (default: 4127)
#line 1019 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.hpp"
retval = (static_cast< long>(max(static_cast< int>(a), static_cast< int>(b)))); 
#line 1020
} else { 
#line 1021
retval = (static_cast< long>(llmax(static_cast< __int64>(a), static_cast< __int64>(b)))); 
#line 1022
}  
#line 1023
return retval; 
#line 1024
} 
#line 1026
static inline unsigned long max(const unsigned long a, const unsigned long b) 
#line 1027
{ 
#line 1028
unsigned long retval; 
#line 1030
#pragma warning (disable: 4127)
#line 1032 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.hpp"
if (sizeof(unsigned long) == sizeof(unsigned)) { 
#line 1034
#pragma warning (default: 4127)
#line 1036 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.hpp"
retval = (static_cast< unsigned long>(umax(static_cast< unsigned>(a), static_cast< unsigned>(b)))); 
#line 1037
} else { 
#line 1038
retval = (static_cast< unsigned long>(ullmax(static_cast< unsigned __int64>(a), static_cast< unsigned __int64>(b)))); 
#line 1039
}  
#line 1040
return retval; 
#line 1041
} 
#line 1043
static inline unsigned long max(const long a, const unsigned long b) 
#line 1044
{ 
#line 1045
unsigned long retval; 
#line 1047
#pragma warning (disable: 4127)
#line 1049 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.hpp"
if (sizeof(unsigned long) == sizeof(unsigned)) { 
#line 1051
#pragma warning (default: 4127)
#line 1053 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.hpp"
retval = (static_cast< unsigned long>(umax(static_cast< unsigned>(a), static_cast< unsigned>(b)))); 
#line 1054
} else { 
#line 1055
retval = (static_cast< unsigned long>(ullmax(static_cast< unsigned __int64>(a), static_cast< unsigned __int64>(b)))); 
#line 1056
}  
#line 1057
return retval; 
#line 1058
} 
#line 1060
static inline unsigned long max(const unsigned long a, const long b) 
#line 1061
{ 
#line 1062
unsigned long retval; 
#line 1064
#pragma warning (disable: 4127)
#line 1066 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.hpp"
if (sizeof(unsigned long) == sizeof(unsigned)) { 
#line 1068
#pragma warning (default: 4127)
#line 1070 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.hpp"
retval = (static_cast< unsigned long>(umax(static_cast< unsigned>(a), static_cast< unsigned>(b)))); 
#line 1071
} else { 
#line 1072
retval = (static_cast< unsigned long>(ullmax(static_cast< unsigned __int64>(a), static_cast< unsigned __int64>(b)))); 
#line 1073
}  
#line 1074
return retval; 
#line 1075
} 
#line 1077
static inline __int64 max(const __int64 a, const __int64 b) 
#line 1078
{ 
#line 1079
return llmax(a, b); 
#line 1080
} 
#line 1082
static inline unsigned __int64 max(const unsigned __int64 a, const unsigned __int64 b) 
#line 1083
{ 
#line 1084
return ullmax(a, b); 
#line 1085
} 
#line 1087
static inline unsigned __int64 max(const __int64 a, const unsigned __int64 b) 
#line 1088
{ 
#line 1089
return ullmax(static_cast< unsigned __int64>(a), b); 
#line 1090
} 
#line 1092
static inline unsigned __int64 max(const unsigned __int64 a, const __int64 b) 
#line 1093
{ 
#line 1094
return ullmax(a, static_cast< unsigned __int64>(b)); 
#line 1095
} 
#line 1097
static inline float max(const float a, const float b) 
#line 1098
{ 
#line 1099
return fmaxf(a, b); 
#line 1100
} 
#line 1102
static inline double max(const double a, const double b) 
#line 1103
{ 
#line 1104
return fmax(a, b); 
#line 1105
} 
#line 1107
static inline double max(const float a, const double b) 
#line 1108
{ 
#line 1109
return fmax(static_cast< double>(a), b); 
#line 1110
} 
#line 1112
static inline double max(const double a, const float b) 
#line 1113
{ 
#line 1114
return fmax(a, static_cast< double>(b)); 
#line 1115
} 
#line 1121
#pragma warning(disable : 4211)
#line 1126 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\math_functions.hpp"
static inline int min(const int a, const int b) 
#line 1127
{ 
#line 1128
return (a < b) ? a : b; 
#line 1129
} 
#line 1131
static inline unsigned umin(const unsigned a, const unsigned b) 
#line 1132
{ 
#line 1133
return (a < b) ? a : b; 
#line 1134
} 
#line 1136
static inline __int64 llmin(const __int64 a, const __int64 b) 
#line 1137
{ 
#line 1138
return (a < b) ? a : b; 
#line 1139
} 
#line 1141
static inline unsigned __int64 ullmin(const unsigned __int64 a, const unsigned __int64 
#line 1142
b) 
#line 1143
{ 
#line 1144
return (a < b) ? a : b; 
#line 1145
} 
#line 1147
static inline int max(const int a, const int b) 
#line 1148
{ 
#line 1149
return (a > b) ? a : b; 
#line 1150
} 
#line 1152
static inline unsigned umax(const unsigned a, const unsigned b) 
#line 1153
{ 
#line 1154
return (a > b) ? a : b; 
#line 1155
} 
#line 1157
static inline __int64 llmax(const __int64 a, const __int64 b) 
#line 1158
{ 
#line 1159
return (a > b) ? a : b; 
#line 1160
} 
#line 1162
static inline unsigned __int64 ullmax(const unsigned __int64 a, const unsigned __int64 
#line 1163
b) 
#line 1164
{ 
#line 1165
return (a > b) ? a : b; 
#line 1166
} 
#line 1169
#pragma warning(default: 4211)
#line 74 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\cuda_surface_types.h"
template< class T, int dim = 1> 
#line 75
struct surface : public surfaceReference { 
#line 78
surface() 
#line 79
{ 
#line 80
(channelDesc) = cudaCreateChannelDesc< T> (); 
#line 81
} 
#line 83
surface(::cudaChannelFormatDesc desc) 
#line 84
{ 
#line 85
(channelDesc) = desc; 
#line 86
} 
#line 88 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\cuda_surface_types.h"
}; 
#line 90
template< int dim> 
#line 91
struct surface< void, dim>  : public surfaceReference { 
#line 94
surface() 
#line 95
{ 
#line 96
(channelDesc) = cudaCreateChannelDesc< void> (); 
#line 97
} 
#line 99 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\cuda_surface_types.h"
}; 
#line 74 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\cuda_texture_types.h"
template< class T, int texType = 1, cudaTextureReadMode mode = cudaReadModeElementType> 
#line 75
struct texture : public textureReference { 
#line 78
texture(int norm = 0, ::cudaTextureFilterMode 
#line 79
fMode = cudaFilterModePoint, ::cudaTextureAddressMode 
#line 80
aMode = cudaAddressModeClamp) 
#line 81
{ 
#line 82
(normalized) = norm; 
#line 83
(filterMode) = fMode; 
#line 84
((addressMode)[0]) = aMode; 
#line 85
((addressMode)[1]) = aMode; 
#line 86
((addressMode)[2]) = aMode; 
#line 87
(channelDesc) = cudaCreateChannelDesc< T> (); 
#line 88
(sRGB) = 0; 
#line 89
} 
#line 91
texture(int norm, ::cudaTextureFilterMode 
#line 92
fMode, ::cudaTextureAddressMode 
#line 93
aMode, ::cudaChannelFormatDesc 
#line 94
desc) 
#line 95
{ 
#line 96
(normalized) = norm; 
#line 97
(filterMode) = fMode; 
#line 98
((addressMode)[0]) = aMode; 
#line 99
((addressMode)[1]) = aMode; 
#line 100
((addressMode)[2]) = aMode; 
#line 101
(channelDesc) = desc; 
#line 102
(sRGB) = 0; 
#line 103
} 
#line 105 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\cuda_texture_types.h"
}; 
#line 89 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt/device_functions.h"
extern "C" {
#line 3207 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt/device_functions.h"
}
#line 3229 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt/device_functions.h"
__declspec(deprecated("mulhi() is deprecated in favor of __mulhi() and may be removed in a future release (Use -Wno-deprecated-declarations to suppress" " this warning).")) static __inline int mulhi(const int a, const int b); 
#line 3231
__declspec(deprecated("mulhi() is deprecated in favor of __mulhi() and may be removed in a future release (Use -Wno-deprecated-declarations to suppress" " this warning).")) static __inline unsigned mulhi(const unsigned a, const unsigned b); 
#line 3233
__declspec(deprecated("mulhi() is deprecated in favor of __mulhi() and may be removed in a future release (Use -Wno-deprecated-declarations to suppress" " this warning).")) static __inline unsigned mulhi(const int a, const unsigned b); 
#line 3235
__declspec(deprecated("mulhi() is deprecated in favor of __mulhi() and may be removed in a future release (Use -Wno-deprecated-declarations to suppress" " this warning).")) static __inline unsigned mulhi(const unsigned a, const int b); 
#line 3237
__declspec(deprecated("mul64hi() is deprecated in favor of __mul64hi() and may be removed in a future release (Use -Wno-deprecated-declarations to supp" "ress this warning).")) static __inline __int64 mul64hi(const __int64 a, const __int64 b); 
#line 3239
__declspec(deprecated("mul64hi() is deprecated in favor of __mul64hi() and may be removed in a future release (Use -Wno-deprecated-declarations to supp" "ress this warning).")) static __inline unsigned __int64 mul64hi(const unsigned __int64 a, const unsigned __int64 b); 
#line 3241
__declspec(deprecated("mul64hi() is deprecated in favor of __mul64hi() and may be removed in a future release (Use -Wno-deprecated-declarations to supp" "ress this warning).")) static __inline unsigned __int64 mul64hi(const __int64 a, const unsigned __int64 b); 
#line 3243
__declspec(deprecated("mul64hi() is deprecated in favor of __mul64hi() and may be removed in a future release (Use -Wno-deprecated-declarations to supp" "ress this warning).")) static __inline unsigned __int64 mul64hi(const unsigned __int64 a, const __int64 b); 
#line 3245
__declspec(deprecated("float_as_int() is deprecated in favor of __float_as_int() and may be removed in a future release (Use -Wno-deprecated-declaratio" "ns to suppress this warning).")) static __inline int float_as_int(const float a); 
#line 3247
__declspec(deprecated("int_as_float() is deprecated in favor of __int_as_float() and may be removed in a future release (Use -Wno-deprecated-declaratio" "ns to suppress this warning).")) static __inline float int_as_float(const int a); 
#line 3249
__declspec(deprecated("float_as_uint() is deprecated in favor of __float_as_uint() and may be removed in a future release (Use -Wno-deprecated-declarat" "ions to suppress this warning).")) static __inline unsigned float_as_uint(const float a); 
#line 3251
__declspec(deprecated("uint_as_float() is deprecated in favor of __uint_as_float() and may be removed in a future release (Use -Wno-deprecated-declarat" "ions to suppress this warning).")) static __inline float uint_as_float(const unsigned a); 
#line 3253
__declspec(deprecated("saturate() is deprecated in favor of __saturatef() and may be removed in a future release (Use -Wno-deprecated-declarations to s" "uppress this warning).")) static __inline float saturate(const float a); 
#line 3255
__declspec(deprecated("mul24() is deprecated in favor of __mul24() and may be removed in a future release (Use -Wno-deprecated-declarations to suppress" " this warning).")) static __inline int mul24(const int a, const int b); 
#line 3257
__declspec(deprecated("umul24() is deprecated in favor of __umul24() and may be removed in a future release (Use -Wno-deprecated-declarations to suppre" "ss this warning).")) static __inline unsigned umul24(const unsigned a, const unsigned b); 
#line 3259
__declspec(deprecated("float2int() is deprecated in favor of __float2int_ru|_rd|_rn|_rz() and may be removed in a future release (Use -Wno-deprecated-d" "eclarations to suppress this warning).")) static __inline int float2int(const float a, const cudaRoundMode mode = cudaRoundZero); 
#line 3261
__declspec(deprecated("float2uint() is deprecated in favor of __float2uint_ru|_rd|_rn|_rz() and may be removed in a future release (Use -Wno-deprecated" "-declarations to suppress this warning).")) static __inline unsigned float2uint(const float a, const cudaRoundMode mode = cudaRoundZero); 
#line 3263
__declspec(deprecated("int2float() is deprecated in favor of __int2float_ru|_rd|_rn|_rz() and may be removed in a future release (Use -Wno-deprecated-d" "eclarations to suppress this warning).")) static __inline float int2float(const int a, const cudaRoundMode mode = cudaRoundNearest); 
#line 3265
__declspec(deprecated("uint2float() is deprecated in favor of __uint2float_ru|_rd|_rn|_rz() and may be removed in a future release (Use -Wno-deprecated" "-declarations to suppress this warning).")) static __inline float uint2float(const unsigned a, const cudaRoundMode mode = cudaRoundNearest); 
#line 90 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\device_functions.hpp"
static __inline int mulhi(const int a, const int b) {int volatile ___ = 1;(void)a;(void)b;::exit(___);}
#if 0
#line 91
{ 
#line 92
return __mulhi(a, b); 
#line 93
} 
#endif
#line 95 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\device_functions.hpp"
static __inline unsigned mulhi(const unsigned a, const unsigned b) {int volatile ___ = 1;(void)a;(void)b;::exit(___);}
#if 0
#line 96
{ 
#line 97
return __umulhi(a, b); 
#line 98
} 
#endif
#line 100 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\device_functions.hpp"
static __inline unsigned mulhi(const int a, const unsigned b) {int volatile ___ = 1;(void)a;(void)b;::exit(___);}
#if 0
#line 101
{ 
#line 102
return __umulhi(static_cast< unsigned>(a), b); 
#line 103
} 
#endif
#line 105 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\device_functions.hpp"
static __inline unsigned mulhi(const unsigned a, const int b) {int volatile ___ = 1;(void)a;(void)b;::exit(___);}
#if 0
#line 106
{ 
#line 107
return __umulhi(a, static_cast< unsigned>(b)); 
#line 108
} 
#endif
#line 110 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\device_functions.hpp"
static __inline __int64 mul64hi(const __int64 a, const __int64 b) {int volatile ___ = 1;(void)a;(void)b;::exit(___);}
#if 0
#line 111
{ 
#line 112
return __mul64hi(a, b); 
#line 113
} 
#endif
#line 115 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\device_functions.hpp"
static __inline unsigned __int64 mul64hi(const unsigned __int64 a, const unsigned __int64 b) {int volatile ___ = 1;(void)a;(void)b;::exit(___);}
#if 0
#line 116
{ 
#line 117
return __umul64hi(a, b); 
#line 118
} 
#endif
#line 120 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\device_functions.hpp"
static __inline unsigned __int64 mul64hi(const __int64 a, const unsigned __int64 b) {int volatile ___ = 1;(void)a;(void)b;::exit(___);}
#if 0
#line 121
{ 
#line 122
return __umul64hi(static_cast< unsigned __int64>(a), b); 
#line 123
} 
#endif
#line 125 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\device_functions.hpp"
static __inline unsigned __int64 mul64hi(const unsigned __int64 a, const __int64 b) {int volatile ___ = 1;(void)a;(void)b;::exit(___);}
#if 0
#line 126
{ 
#line 127
return __umul64hi(a, static_cast< unsigned __int64>(b)); 
#line 128
} 
#endif
#line 130 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\device_functions.hpp"
static __inline int float_as_int(const float a) {int volatile ___ = 1;(void)a;::exit(___);}
#if 0
#line 131
{ 
#line 132
return __float_as_int(a); 
#line 133
} 
#endif
#line 135 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\device_functions.hpp"
static __inline float int_as_float(const int a) {int volatile ___ = 1;(void)a;::exit(___);}
#if 0
#line 136
{ 
#line 137
return __int_as_float(a); 
#line 138
} 
#endif
#line 140 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\device_functions.hpp"
static __inline unsigned float_as_uint(const float a) {int volatile ___ = 1;(void)a;::exit(___);}
#if 0
#line 141
{ 
#line 142
return __float_as_uint(a); 
#line 143
} 
#endif
#line 145 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\device_functions.hpp"
static __inline float uint_as_float(const unsigned a) {int volatile ___ = 1;(void)a;::exit(___);}
#if 0
#line 146
{ 
#line 147
return __uint_as_float(a); 
#line 148
} 
#endif
#line 149 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\device_functions.hpp"
static __inline float saturate(const float a) {int volatile ___ = 1;(void)a;::exit(___);}
#if 0
#line 150
{ 
#line 151
return __saturatef(a); 
#line 152
} 
#endif
#line 154 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\device_functions.hpp"
static __inline int mul24(const int a, const int b) {int volatile ___ = 1;(void)a;(void)b;::exit(___);}
#if 0
#line 155
{ 
#line 156
return __mul24(a, b); 
#line 157
} 
#endif
#line 159 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\device_functions.hpp"
static __inline unsigned umul24(const unsigned a, const unsigned b) {int volatile ___ = 1;(void)a;(void)b;::exit(___);}
#if 0
#line 160
{ 
#line 161
return __umul24(a, b); 
#line 162
} 
#endif
#line 164 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\device_functions.hpp"
static __inline int float2int(const float a, const cudaRoundMode mode) {int volatile ___ = 1;(void)a;(void)mode;::exit(___);}
#if 0
#line 165
{ 
#line 166
return (mode == (cudaRoundNearest)) ? __float2int_rn(a) : ((mode == (cudaRoundPosInf)) ? __float2int_ru(a) : ((mode == (cudaRoundMinInf)) ? __float2int_rd(a) : __float2int_rz(a))); 
#line 170
} 
#endif
#line 172 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\device_functions.hpp"
static __inline unsigned float2uint(const float a, const cudaRoundMode mode) {int volatile ___ = 1;(void)a;(void)mode;::exit(___);}
#if 0
#line 173
{ 
#line 174
return (mode == (cudaRoundNearest)) ? __float2uint_rn(a) : ((mode == (cudaRoundPosInf)) ? __float2uint_ru(a) : ((mode == (cudaRoundMinInf)) ? __float2uint_rd(a) : __float2uint_rz(a))); 
#line 178
} 
#endif
#line 180 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\device_functions.hpp"
static __inline float int2float(const int a, const cudaRoundMode mode) {int volatile ___ = 1;(void)a;(void)mode;::exit(___);}
#if 0
#line 181
{ 
#line 182
return (mode == (cudaRoundZero)) ? __int2float_rz(a) : ((mode == (cudaRoundPosInf)) ? __int2float_ru(a) : ((mode == (cudaRoundMinInf)) ? __int2float_rd(a) : __int2float_rn(a))); 
#line 186
} 
#endif
#line 188 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\device_functions.hpp"
static __inline float uint2float(const unsigned a, const cudaRoundMode mode) {int volatile ___ = 1;(void)a;(void)mode;::exit(___);}
#if 0
#line 189
{ 
#line 190
return (mode == (cudaRoundZero)) ? __uint2float_rz(a) : ((mode == (cudaRoundPosInf)) ? __uint2float_ru(a) : ((mode == (cudaRoundMinInf)) ? __uint2float_rd(a) : __uint2float_rn(a))); 
#line 194
} 
#endif
#line 106 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\device_atomic_functions.h"
static __inline int atomicAdd(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 106
{ } 
#endif
#line 108 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\device_atomic_functions.h"
static __inline unsigned atomicAdd(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 108
{ } 
#endif
#line 110 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\device_atomic_functions.h"
static __inline int atomicSub(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 110
{ } 
#endif
#line 112 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\device_atomic_functions.h"
static __inline unsigned atomicSub(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 112
{ } 
#endif
#line 114 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\device_atomic_functions.h"
static __inline int atomicExch(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 114
{ } 
#endif
#line 116 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\device_atomic_functions.h"
static __inline unsigned atomicExch(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 116
{ } 
#endif
#line 118 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\device_atomic_functions.h"
static __inline float atomicExch(float *address, float val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 118
{ } 
#endif
#line 120 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\device_atomic_functions.h"
static __inline int atomicMin(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 120
{ } 
#endif
#line 122 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\device_atomic_functions.h"
static __inline unsigned atomicMin(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 122
{ } 
#endif
#line 124 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\device_atomic_functions.h"
static __inline int atomicMax(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 124
{ } 
#endif
#line 126 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\device_atomic_functions.h"
static __inline unsigned atomicMax(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 126
{ } 
#endif
#line 128 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\device_atomic_functions.h"
static __inline unsigned atomicInc(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 128
{ } 
#endif
#line 130 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\device_atomic_functions.h"
static __inline unsigned atomicDec(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 130
{ } 
#endif
#line 132 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\device_atomic_functions.h"
static __inline int atomicAnd(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 132
{ } 
#endif
#line 134 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\device_atomic_functions.h"
static __inline unsigned atomicAnd(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 134
{ } 
#endif
#line 136 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\device_atomic_functions.h"
static __inline int atomicOr(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 136
{ } 
#endif
#line 138 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\device_atomic_functions.h"
static __inline unsigned atomicOr(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 138
{ } 
#endif
#line 140 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\device_atomic_functions.h"
static __inline int atomicXor(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 140
{ } 
#endif
#line 142 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\device_atomic_functions.h"
static __inline unsigned atomicXor(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 142
{ } 
#endif
#line 144 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\device_atomic_functions.h"
static __inline int atomicCAS(int *address, int compare, int val) {int volatile ___ = 1;(void)address;(void)compare;(void)val;::exit(___);}
#if 0
#line 144
{ } 
#endif
#line 146 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\device_atomic_functions.h"
static __inline unsigned atomicCAS(unsigned *address, unsigned compare, unsigned val) {int volatile ___ = 1;(void)address;(void)compare;(void)val;::exit(___);}
#if 0
#line 146
{ } 
#endif
#line 171 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\device_atomic_functions.h"
extern "C" {
#line 180
}
#line 189
static __inline unsigned __int64 atomicAdd(unsigned __int64 *address, unsigned __int64 val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 189
{ } 
#endif
#line 191 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\device_atomic_functions.h"
static __inline unsigned __int64 atomicExch(unsigned __int64 *address, unsigned __int64 val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 191
{ } 
#endif
#line 193 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\device_atomic_functions.h"
static __inline unsigned __int64 atomicCAS(unsigned __int64 *address, unsigned __int64 compare, unsigned __int64 val) {int volatile ___ = 1;(void)address;(void)compare;(void)val;::exit(___);}
#if 0
#line 193
{ } 
#endif
#line 195 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\device_atomic_functions.h"
__declspec(deprecated("__any() is deprecated in favor of __any_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to suppr" "ess this warning).")) static __inline bool any(bool cond) {int volatile ___ = 1;(void)cond;::exit(___);}
#if 0
#line 195
{ } 
#endif
#line 197 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\device_atomic_functions.h"
__declspec(deprecated("__all() is deprecated in favor of __all_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to suppr" "ess this warning).")) static __inline bool all(bool cond) {int volatile ___ = 1;(void)cond;::exit(___);}
#if 0
#line 197
{ } 
#endif
#line 87 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\device_double_functions.h"
extern "C" {
#line 1139
}
#line 1147
static __inline double fma(double a, double b, double c, cudaRoundMode mode); 
#line 1149
static __inline double dmul(double a, double b, cudaRoundMode mode = cudaRoundNearest); 
#line 1151
static __inline double dadd(double a, double b, cudaRoundMode mode = cudaRoundNearest); 
#line 1153
static __inline double dsub(double a, double b, cudaRoundMode mode = cudaRoundNearest); 
#line 1155
static __inline int double2int(double a, cudaRoundMode mode = cudaRoundZero); 
#line 1157
static __inline unsigned double2uint(double a, cudaRoundMode mode = cudaRoundZero); 
#line 1159
static __inline __int64 double2ll(double a, cudaRoundMode mode = cudaRoundZero); 
#line 1161
static __inline unsigned __int64 double2ull(double a, cudaRoundMode mode = cudaRoundZero); 
#line 1163
static __inline double ll2double(__int64 a, cudaRoundMode mode = cudaRoundNearest); 
#line 1165
static __inline double ull2double(unsigned __int64 a, cudaRoundMode mode = cudaRoundNearest); 
#line 1167
static __inline double int2double(int a, cudaRoundMode mode = cudaRoundNearest); 
#line 1169
static __inline double uint2double(unsigned a, cudaRoundMode mode = cudaRoundNearest); 
#line 1171
static __inline double float2double(float a, cudaRoundMode mode = cudaRoundNearest); 
#line 93 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\device_double_functions.hpp"
static __inline double fma(double a, double b, double c, cudaRoundMode mode) {int volatile ___ = 1;(void)a;(void)b;(void)c;(void)mode;::exit(___);}
#if 0
#line 94
{ 
#line 95
return (mode == (cudaRoundZero)) ? __fma_rz(a, b, c) : ((mode == (cudaRoundPosInf)) ? __fma_ru(a, b, c) : ((mode == (cudaRoundMinInf)) ? __fma_rd(a, b, c) : __fma_rn(a, b, c))); 
#line 99
} 
#endif
#line 101 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\device_double_functions.hpp"
static __inline double dmul(double a, double b, cudaRoundMode mode) {int volatile ___ = 1;(void)a;(void)b;(void)mode;::exit(___);}
#if 0
#line 102
{ 
#line 103
return (mode == (cudaRoundZero)) ? __dmul_rz(a, b) : ((mode == (cudaRoundPosInf)) ? __dmul_ru(a, b) : ((mode == (cudaRoundMinInf)) ? __dmul_rd(a, b) : __dmul_rn(a, b))); 
#line 107
} 
#endif
#line 109 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\device_double_functions.hpp"
static __inline double dadd(double a, double b, cudaRoundMode mode) {int volatile ___ = 1;(void)a;(void)b;(void)mode;::exit(___);}
#if 0
#line 110
{ 
#line 111
return (mode == (cudaRoundZero)) ? __dadd_rz(a, b) : ((mode == (cudaRoundPosInf)) ? __dadd_ru(a, b) : ((mode == (cudaRoundMinInf)) ? __dadd_rd(a, b) : __dadd_rn(a, b))); 
#line 115
} 
#endif
#line 117 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\device_double_functions.hpp"
static __inline double dsub(double a, double b, cudaRoundMode mode) {int volatile ___ = 1;(void)a;(void)b;(void)mode;::exit(___);}
#if 0
#line 118
{ 
#line 119
return (mode == (cudaRoundZero)) ? __dsub_rz(a, b) : ((mode == (cudaRoundPosInf)) ? __dsub_ru(a, b) : ((mode == (cudaRoundMinInf)) ? __dsub_rd(a, b) : __dsub_rn(a, b))); 
#line 123
} 
#endif
#line 125 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\device_double_functions.hpp"
static __inline int double2int(double a, cudaRoundMode mode) {int volatile ___ = 1;(void)a;(void)mode;::exit(___);}
#if 0
#line 126
{ 
#line 127
return (mode == (cudaRoundNearest)) ? __double2int_rn(a) : ((mode == (cudaRoundPosInf)) ? __double2int_ru(a) : ((mode == (cudaRoundMinInf)) ? __double2int_rd(a) : __double2int_rz(a))); 
#line 131
} 
#endif
#line 133 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\device_double_functions.hpp"
static __inline unsigned double2uint(double a, cudaRoundMode mode) {int volatile ___ = 1;(void)a;(void)mode;::exit(___);}
#if 0
#line 134
{ 
#line 135
return (mode == (cudaRoundNearest)) ? __double2uint_rn(a) : ((mode == (cudaRoundPosInf)) ? __double2uint_ru(a) : ((mode == (cudaRoundMinInf)) ? __double2uint_rd(a) : __double2uint_rz(a))); 
#line 139
} 
#endif
#line 141 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\device_double_functions.hpp"
static __inline __int64 double2ll(double a, cudaRoundMode mode) {int volatile ___ = 1;(void)a;(void)mode;::exit(___);}
#if 0
#line 142
{ 
#line 143
return (mode == (cudaRoundNearest)) ? __double2ll_rn(a) : ((mode == (cudaRoundPosInf)) ? __double2ll_ru(a) : ((mode == (cudaRoundMinInf)) ? __double2ll_rd(a) : __double2ll_rz(a))); 
#line 147
} 
#endif
#line 149 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\device_double_functions.hpp"
static __inline unsigned __int64 double2ull(double a, cudaRoundMode mode) {int volatile ___ = 1;(void)a;(void)mode;::exit(___);}
#if 0
#line 150
{ 
#line 151
return (mode == (cudaRoundNearest)) ? __double2ull_rn(a) : ((mode == (cudaRoundPosInf)) ? __double2ull_ru(a) : ((mode == (cudaRoundMinInf)) ? __double2ull_rd(a) : __double2ull_rz(a))); 
#line 155
} 
#endif
#line 157 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\device_double_functions.hpp"
static __inline double ll2double(__int64 a, cudaRoundMode mode) {int volatile ___ = 1;(void)a;(void)mode;::exit(___);}
#if 0
#line 158
{ 
#line 159
return (mode == (cudaRoundZero)) ? __ll2double_rz(a) : ((mode == (cudaRoundPosInf)) ? __ll2double_ru(a) : ((mode == (cudaRoundMinInf)) ? __ll2double_rd(a) : __ll2double_rn(a))); 
#line 163
} 
#endif
#line 165 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\device_double_functions.hpp"
static __inline double ull2double(unsigned __int64 a, cudaRoundMode mode) {int volatile ___ = 1;(void)a;(void)mode;::exit(___);}
#if 0
#line 166
{ 
#line 167
return (mode == (cudaRoundZero)) ? __ull2double_rz(a) : ((mode == (cudaRoundPosInf)) ? __ull2double_ru(a) : ((mode == (cudaRoundMinInf)) ? __ull2double_rd(a) : __ull2double_rn(a))); 
#line 171
} 
#endif
#line 173 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\device_double_functions.hpp"
static __inline double int2double(int a, cudaRoundMode mode) {int volatile ___ = 1;(void)a;(void)mode;::exit(___);}
#if 0
#line 174
{ 
#line 175
return (double)a; 
#line 176
} 
#endif
#line 178 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\device_double_functions.hpp"
static __inline double uint2double(unsigned a, cudaRoundMode mode) {int volatile ___ = 1;(void)a;(void)mode;::exit(___);}
#if 0
#line 179
{ 
#line 180
return (double)a; 
#line 181
} 
#endif
#line 183 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\device_double_functions.hpp"
static __inline double float2double(float a, cudaRoundMode mode) {int volatile ___ = 1;(void)a;(void)mode;::exit(___);}
#if 0
#line 184
{ 
#line 185
return (double)a; 
#line 186
} 
#endif
#line 89 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_20_atomic_functions.h"
static __inline float atomicAdd(float *address, float val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 89
{ } 
#endif
#line 100 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_atomic_functions.h"
static __inline __int64 atomicMin(__int64 *address, __int64 val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 100
{ } 
#endif
#line 102 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_atomic_functions.h"
static __inline __int64 atomicMax(__int64 *address, __int64 val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 102
{ } 
#endif
#line 104 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_atomic_functions.h"
static __inline __int64 atomicAnd(__int64 *address, __int64 val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 104
{ } 
#endif
#line 106 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_atomic_functions.h"
static __inline __int64 atomicOr(__int64 *address, __int64 val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 106
{ } 
#endif
#line 108 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_atomic_functions.h"
static __inline __int64 atomicXor(__int64 *address, __int64 val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 108
{ } 
#endif
#line 110 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_atomic_functions.h"
static __inline unsigned __int64 atomicMin(unsigned __int64 *address, unsigned __int64 val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 110
{ } 
#endif
#line 112 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_atomic_functions.h"
static __inline unsigned __int64 atomicMax(unsigned __int64 *address, unsigned __int64 val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 112
{ } 
#endif
#line 114 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_atomic_functions.h"
static __inline unsigned __int64 atomicAnd(unsigned __int64 *address, unsigned __int64 val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 114
{ } 
#endif
#line 116 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_atomic_functions.h"
static __inline unsigned __int64 atomicOr(unsigned __int64 *address, unsigned __int64 val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 116
{ } 
#endif
#line 118 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_atomic_functions.h"
static __inline unsigned __int64 atomicXor(unsigned __int64 *address, unsigned __int64 val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 118
{ } 
#endif
#line 303 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_60_atomic_functions.h"
static __inline double atomicAdd(double *address, double val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 303
{ } 
#endif
#line 306 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_60_atomic_functions.h"
static __inline int atomicAdd_block(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 306
{ } 
#endif
#line 309 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_60_atomic_functions.h"
static __inline int atomicAdd_system(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 309
{ } 
#endif
#line 312 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_60_atomic_functions.h"
static __inline unsigned atomicAdd_block(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 312
{ } 
#endif
#line 315 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_60_atomic_functions.h"
static __inline unsigned atomicAdd_system(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 315
{ } 
#endif
#line 318 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_60_atomic_functions.h"
static __inline unsigned __int64 atomicAdd_block(unsigned __int64 *address, unsigned __int64 val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 318
{ } 
#endif
#line 321 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_60_atomic_functions.h"
static __inline unsigned __int64 atomicAdd_system(unsigned __int64 *address, unsigned __int64 val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 321
{ } 
#endif
#line 324 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_60_atomic_functions.h"
static __inline float atomicAdd_block(float *address, float val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 324
{ } 
#endif
#line 327 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_60_atomic_functions.h"
static __inline float atomicAdd_system(float *address, float val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 327
{ } 
#endif
#line 330 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_60_atomic_functions.h"
static __inline double atomicAdd_block(double *address, double val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 330
{ } 
#endif
#line 333 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_60_atomic_functions.h"
static __inline double atomicAdd_system(double *address, double val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 333
{ } 
#endif
#line 336 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_60_atomic_functions.h"
static __inline int atomicSub_block(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 336
{ } 
#endif
#line 339 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_60_atomic_functions.h"
static __inline int atomicSub_system(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 339
{ } 
#endif
#line 342 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_60_atomic_functions.h"
static __inline unsigned atomicSub_block(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 342
{ } 
#endif
#line 345 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_60_atomic_functions.h"
static __inline unsigned atomicSub_system(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 345
{ } 
#endif
#line 348 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_60_atomic_functions.h"
static __inline int atomicExch_block(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 348
{ } 
#endif
#line 351 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_60_atomic_functions.h"
static __inline int atomicExch_system(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 351
{ } 
#endif
#line 354 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_60_atomic_functions.h"
static __inline unsigned atomicExch_block(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 354
{ } 
#endif
#line 357 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_60_atomic_functions.h"
static __inline unsigned atomicExch_system(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 357
{ } 
#endif
#line 360 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_60_atomic_functions.h"
static __inline unsigned __int64 atomicExch_block(unsigned __int64 *address, unsigned __int64 val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 360
{ } 
#endif
#line 363 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_60_atomic_functions.h"
static __inline unsigned __int64 atomicExch_system(unsigned __int64 *address, unsigned __int64 val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 363
{ } 
#endif
#line 366 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_60_atomic_functions.h"
static __inline float atomicExch_block(float *address, float val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 366
{ } 
#endif
#line 369 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_60_atomic_functions.h"
static __inline float atomicExch_system(float *address, float val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 369
{ } 
#endif
#line 372 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_60_atomic_functions.h"
static __inline int atomicMin_block(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 372
{ } 
#endif
#line 375 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_60_atomic_functions.h"
static __inline int atomicMin_system(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 375
{ } 
#endif
#line 378 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_60_atomic_functions.h"
static __inline __int64 atomicMin_block(__int64 *address, __int64 val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 378
{ } 
#endif
#line 381 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_60_atomic_functions.h"
static __inline __int64 atomicMin_system(__int64 *address, __int64 val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 381
{ } 
#endif
#line 384 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_60_atomic_functions.h"
static __inline unsigned atomicMin_block(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 384
{ } 
#endif
#line 387 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_60_atomic_functions.h"
static __inline unsigned atomicMin_system(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 387
{ } 
#endif
#line 390 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_60_atomic_functions.h"
static __inline unsigned __int64 atomicMin_block(unsigned __int64 *address, unsigned __int64 val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 390
{ } 
#endif
#line 393 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_60_atomic_functions.h"
static __inline unsigned __int64 atomicMin_system(unsigned __int64 *address, unsigned __int64 val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 393
{ } 
#endif
#line 396 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_60_atomic_functions.h"
static __inline int atomicMax_block(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 396
{ } 
#endif
#line 399 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_60_atomic_functions.h"
static __inline int atomicMax_system(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 399
{ } 
#endif
#line 402 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_60_atomic_functions.h"
static __inline __int64 atomicMax_block(__int64 *address, __int64 val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 402
{ } 
#endif
#line 405 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_60_atomic_functions.h"
static __inline __int64 atomicMax_system(__int64 *address, __int64 val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 405
{ } 
#endif
#line 408 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_60_atomic_functions.h"
static __inline unsigned atomicMax_block(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 408
{ } 
#endif
#line 411 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_60_atomic_functions.h"
static __inline unsigned atomicMax_system(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 411
{ } 
#endif
#line 414 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_60_atomic_functions.h"
static __inline unsigned __int64 atomicMax_block(unsigned __int64 *address, unsigned __int64 val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 414
{ } 
#endif
#line 417 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_60_atomic_functions.h"
static __inline unsigned __int64 atomicMax_system(unsigned __int64 *address, unsigned __int64 val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 417
{ } 
#endif
#line 420 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_60_atomic_functions.h"
static __inline unsigned atomicInc_block(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 420
{ } 
#endif
#line 423 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_60_atomic_functions.h"
static __inline unsigned atomicInc_system(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 423
{ } 
#endif
#line 426 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_60_atomic_functions.h"
static __inline unsigned atomicDec_block(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 426
{ } 
#endif
#line 429 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_60_atomic_functions.h"
static __inline unsigned atomicDec_system(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 429
{ } 
#endif
#line 432 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_60_atomic_functions.h"
static __inline int atomicCAS_block(int *address, int compare, int val) {int volatile ___ = 1;(void)address;(void)compare;(void)val;::exit(___);}
#if 0
#line 432
{ } 
#endif
#line 435 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_60_atomic_functions.h"
static __inline int atomicCAS_system(int *address, int compare, int val) {int volatile ___ = 1;(void)address;(void)compare;(void)val;::exit(___);}
#if 0
#line 435
{ } 
#endif
#line 438 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_60_atomic_functions.h"
static __inline unsigned atomicCAS_block(unsigned *address, unsigned compare, unsigned 
#line 439
val) {int volatile ___ = 1;(void)address;(void)compare;(void)val;::exit(___);}
#if 0
#line 439
{ } 
#endif
#line 442 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_60_atomic_functions.h"
static __inline unsigned atomicCAS_system(unsigned *address, unsigned compare, unsigned 
#line 443
val) {int volatile ___ = 1;(void)address;(void)compare;(void)val;::exit(___);}
#if 0
#line 443
{ } 
#endif
#line 446 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_60_atomic_functions.h"
static __inline unsigned __int64 atomicCAS_block(unsigned __int64 *address, unsigned __int64 
#line 447
compare, unsigned __int64 
#line 448
val) {int volatile ___ = 1;(void)address;(void)compare;(void)val;::exit(___);}
#if 0
#line 448
{ } 
#endif
#line 451 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_60_atomic_functions.h"
static __inline unsigned __int64 atomicCAS_system(unsigned __int64 *address, unsigned __int64 
#line 452
compare, unsigned __int64 
#line 453
val) {int volatile ___ = 1;(void)address;(void)compare;(void)val;::exit(___);}
#if 0
#line 453
{ } 
#endif
#line 456 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_60_atomic_functions.h"
static __inline int atomicAnd_block(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 456
{ } 
#endif
#line 459 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_60_atomic_functions.h"
static __inline int atomicAnd_system(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 459
{ } 
#endif
#line 462 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_60_atomic_functions.h"
static __inline __int64 atomicAnd_block(__int64 *address, __int64 val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 462
{ } 
#endif
#line 465 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_60_atomic_functions.h"
static __inline __int64 atomicAnd_system(__int64 *address, __int64 val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 465
{ } 
#endif
#line 468 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_60_atomic_functions.h"
static __inline unsigned atomicAnd_block(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 468
{ } 
#endif
#line 471 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_60_atomic_functions.h"
static __inline unsigned atomicAnd_system(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 471
{ } 
#endif
#line 474 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_60_atomic_functions.h"
static __inline unsigned __int64 atomicAnd_block(unsigned __int64 *address, unsigned __int64 val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 474
{ } 
#endif
#line 477 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_60_atomic_functions.h"
static __inline unsigned __int64 atomicAnd_system(unsigned __int64 *address, unsigned __int64 val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 477
{ } 
#endif
#line 480 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_60_atomic_functions.h"
static __inline int atomicOr_block(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 480
{ } 
#endif
#line 483 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_60_atomic_functions.h"
static __inline int atomicOr_system(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 483
{ } 
#endif
#line 486 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_60_atomic_functions.h"
static __inline __int64 atomicOr_block(__int64 *address, __int64 val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 486
{ } 
#endif
#line 489 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_60_atomic_functions.h"
static __inline __int64 atomicOr_system(__int64 *address, __int64 val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 489
{ } 
#endif
#line 492 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_60_atomic_functions.h"
static __inline unsigned atomicOr_block(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 492
{ } 
#endif
#line 495 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_60_atomic_functions.h"
static __inline unsigned atomicOr_system(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 495
{ } 
#endif
#line 498 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_60_atomic_functions.h"
static __inline unsigned __int64 atomicOr_block(unsigned __int64 *address, unsigned __int64 val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 498
{ } 
#endif
#line 501 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_60_atomic_functions.h"
static __inline unsigned __int64 atomicOr_system(unsigned __int64 *address, unsigned __int64 val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 501
{ } 
#endif
#line 504 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_60_atomic_functions.h"
static __inline int atomicXor_block(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 504
{ } 
#endif
#line 507 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_60_atomic_functions.h"
static __inline int atomicXor_system(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 507
{ } 
#endif
#line 510 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_60_atomic_functions.h"
static __inline __int64 atomicXor_block(__int64 *address, __int64 val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 510
{ } 
#endif
#line 513 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_60_atomic_functions.h"
static __inline __int64 atomicXor_system(__int64 *address, __int64 val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 513
{ } 
#endif
#line 516 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_60_atomic_functions.h"
static __inline unsigned atomicXor_block(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 516
{ } 
#endif
#line 519 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_60_atomic_functions.h"
static __inline unsigned atomicXor_system(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 519
{ } 
#endif
#line 522 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_60_atomic_functions.h"
static __inline unsigned __int64 atomicXor_block(unsigned __int64 *address, unsigned __int64 val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 522
{ } 
#endif
#line 525 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_60_atomic_functions.h"
static __inline unsigned __int64 atomicXor_system(unsigned __int64 *address, unsigned __int64 val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 525
{ } 
#endif
#line 90 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_20_intrinsics.h"
extern "C" {
#line 1503
}
#line 1510
__declspec(deprecated("__ballot() is deprecated in favor of __ballot_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to" " suppress this warning).")) static __inline unsigned ballot(bool pred) {int volatile ___ = 1;(void)pred;::exit(___);}
#if 0
#line 1510
{ } 
#endif
#line 1512 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_20_intrinsics.h"
static __inline int syncthreads_count(bool pred) {int volatile ___ = 1;(void)pred;::exit(___);}
#if 0
#line 1512
{ } 
#endif
#line 1514 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_20_intrinsics.h"
static __inline bool syncthreads_and(bool pred) {int volatile ___ = 1;(void)pred;::exit(___);}
#if 0
#line 1514
{ } 
#endif
#line 1516 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_20_intrinsics.h"
static __inline bool syncthreads_or(bool pred) {int volatile ___ = 1;(void)pred;::exit(___);}
#if 0
#line 1516
{ } 
#endif
#line 1521 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_20_intrinsics.h"
static __inline unsigned __isGlobal(const void *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 1521
{ } 
#endif
#line 1522 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_20_intrinsics.h"
static __inline unsigned __isShared(const void *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 1522
{ } 
#endif
#line 1523 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_20_intrinsics.h"
static __inline unsigned __isConstant(const void *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 1523
{ } 
#endif
#line 1524 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_20_intrinsics.h"
static __inline unsigned __isLocal(const void *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 1524
{ } 
#endif
#line 1526 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_20_intrinsics.h"
static __inline unsigned __isGridConstant(const void *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 1526
{ } 
#endif
#line 1528 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_20_intrinsics.h"
static __inline size_t __cvta_generic_to_global(const void *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 1528
{ } 
#endif
#line 1529 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_20_intrinsics.h"
static __inline size_t __cvta_generic_to_shared(const void *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 1529
{ } 
#endif
#line 1530 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_20_intrinsics.h"
static __inline size_t __cvta_generic_to_constant(const void *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 1530
{ } 
#endif
#line 1531 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_20_intrinsics.h"
static __inline size_t __cvta_generic_to_local(const void *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 1531
{ } 
#endif
#line 1533 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_20_intrinsics.h"
static __inline size_t __cvta_generic_to_grid_constant(const void *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 1533
{ } 
#endif
#line 1536 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_20_intrinsics.h"
static __inline void *__cvta_global_to_generic(size_t rawbits) {int volatile ___ = 1;(void)rawbits;::exit(___);}
#if 0
#line 1536
{ } 
#endif
#line 1537 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_20_intrinsics.h"
static __inline void *__cvta_shared_to_generic(size_t rawbits) {int volatile ___ = 1;(void)rawbits;::exit(___);}
#if 0
#line 1537
{ } 
#endif
#line 1538 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_20_intrinsics.h"
static __inline void *__cvta_constant_to_generic(size_t rawbits) {int volatile ___ = 1;(void)rawbits;::exit(___);}
#if 0
#line 1538
{ } 
#endif
#line 1539 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_20_intrinsics.h"
static __inline void *__cvta_local_to_generic(size_t rawbits) {int volatile ___ = 1;(void)rawbits;::exit(___);}
#if 0
#line 1539
{ } 
#endif
#line 1541 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_20_intrinsics.h"
static __inline void *__cvta_grid_constant_to_generic(size_t rawbits) {int volatile ___ = 1;(void)rawbits;::exit(___);}
#if 0
#line 1541
{ } 
#endif
#line 102 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_30_intrinsics.h"
static __inline unsigned __fns(unsigned mask, unsigned base, int offset) {int volatile ___ = 1;(void)mask;(void)base;(void)offset;::exit(___);}
#if 0
#line 102
{ } 
#endif
#line 103 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_30_intrinsics.h"
static __inline void __barrier_sync(unsigned id) {int volatile ___ = 1;(void)id;::exit(___);}
#if 0
#line 103
{ } 
#endif
#line 104 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_30_intrinsics.h"
static __inline void __barrier_sync_count(unsigned id, unsigned cnt) {int volatile ___ = 1;(void)id;(void)cnt;::exit(___);}
#if 0
#line 104
{ } 
#endif
#line 105 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_30_intrinsics.h"
static __inline void __syncwarp(unsigned mask = 4294967295U) {int volatile ___ = 1;(void)mask;::exit(___);}
#if 0
#line 105
{ } 
#endif
#line 106 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_30_intrinsics.h"
static __inline int __all_sync(unsigned mask, int pred) {int volatile ___ = 1;(void)mask;(void)pred;::exit(___);}
#if 0
#line 106
{ } 
#endif
#line 107 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_30_intrinsics.h"
static __inline int __any_sync(unsigned mask, int pred) {int volatile ___ = 1;(void)mask;(void)pred;::exit(___);}
#if 0
#line 107
{ } 
#endif
#line 108 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_30_intrinsics.h"
static __inline int __uni_sync(unsigned mask, int pred) {int volatile ___ = 1;(void)mask;(void)pred;::exit(___);}
#if 0
#line 108
{ } 
#endif
#line 109 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_30_intrinsics.h"
static __inline unsigned __ballot_sync(unsigned mask, int pred) {int volatile ___ = 1;(void)mask;(void)pred;::exit(___);}
#if 0
#line 109
{ } 
#endif
#line 110 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_30_intrinsics.h"
static __inline unsigned __activemask() {int volatile ___ = 1;::exit(___);}
#if 0
#line 110
{ } 
#endif
#line 119 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_30_intrinsics.h"
__declspec(deprecated("__shfl() is deprecated in favor of __shfl_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to sup" "press this warning).")) static __inline int __shfl(int var, int srcLane, int width = 32) {int volatile ___ = 1;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
#line 119
{ } 
#endif
#line 120 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_30_intrinsics.h"
__declspec(deprecated("__shfl() is deprecated in favor of __shfl_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to sup" "press this warning).")) static __inline unsigned __shfl(unsigned var, int srcLane, int width = 32) {int volatile ___ = 1;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
#line 120
{ } 
#endif
#line 121 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_30_intrinsics.h"
__declspec(deprecated("__shfl_up() is deprecated in favor of __shfl_up_sync() and may be removed in a future release (Use -Wno-deprecated-declarations " "to suppress this warning).")) static __inline int __shfl_up(int var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 121
{ } 
#endif
#line 122 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_30_intrinsics.h"
__declspec(deprecated("__shfl_up() is deprecated in favor of __shfl_up_sync() and may be removed in a future release (Use -Wno-deprecated-declarations " "to suppress this warning).")) static __inline unsigned __shfl_up(unsigned var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 122
{ } 
#endif
#line 123 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_30_intrinsics.h"
__declspec(deprecated("__shfl_down() is deprecated in favor of __shfl_down_sync() and may be removed in a future release (Use -Wno-deprecated-declarati" "ons to suppress this warning).")) static __inline int __shfl_down(int var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 123
{ } 
#endif
#line 124 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_30_intrinsics.h"
__declspec(deprecated("__shfl_down() is deprecated in favor of __shfl_down_sync() and may be removed in a future release (Use -Wno-deprecated-declarati" "ons to suppress this warning).")) static __inline unsigned __shfl_down(unsigned var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 124
{ } 
#endif
#line 125 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_30_intrinsics.h"
__declspec(deprecated("__shfl_xor() is deprecated in favor of __shfl_xor_sync() and may be removed in a future release (Use -Wno-deprecated-declaration" "s to suppress this warning).")) static __inline int __shfl_xor(int var, int laneMask, int width = 32) {int volatile ___ = 1;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
#line 125
{ } 
#endif
#line 126 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_30_intrinsics.h"
__declspec(deprecated("__shfl_xor() is deprecated in favor of __shfl_xor_sync() and may be removed in a future release (Use -Wno-deprecated-declaration" "s to suppress this warning).")) static __inline unsigned __shfl_xor(unsigned var, int laneMask, int width = 32) {int volatile ___ = 1;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
#line 126
{ } 
#endif
#line 127 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_30_intrinsics.h"
__declspec(deprecated("__shfl() is deprecated in favor of __shfl_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to sup" "press this warning).")) static __inline float __shfl(float var, int srcLane, int width = 32) {int volatile ___ = 1;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
#line 127
{ } 
#endif
#line 128 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_30_intrinsics.h"
__declspec(deprecated("__shfl_up() is deprecated in favor of __shfl_up_sync() and may be removed in a future release (Use -Wno-deprecated-declarations " "to suppress this warning).")) static __inline float __shfl_up(float var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 128
{ } 
#endif
#line 129 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_30_intrinsics.h"
__declspec(deprecated("__shfl_down() is deprecated in favor of __shfl_down_sync() and may be removed in a future release (Use -Wno-deprecated-declarati" "ons to suppress this warning).")) static __inline float __shfl_down(float var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 129
{ } 
#endif
#line 130 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_30_intrinsics.h"
__declspec(deprecated("__shfl_xor() is deprecated in favor of __shfl_xor_sync() and may be removed in a future release (Use -Wno-deprecated-declaration" "s to suppress this warning).")) static __inline float __shfl_xor(float var, int laneMask, int width = 32) {int volatile ___ = 1;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
#line 130
{ } 
#endif
#line 133 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_30_intrinsics.h"
static __inline int __shfl_sync(unsigned mask, int var, int srcLane, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
#line 133
{ } 
#endif
#line 134 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_30_intrinsics.h"
static __inline unsigned __shfl_sync(unsigned mask, unsigned var, int srcLane, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
#line 134
{ } 
#endif
#line 135 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_30_intrinsics.h"
static __inline int __shfl_up_sync(unsigned mask, int var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 135
{ } 
#endif
#line 136 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_30_intrinsics.h"
static __inline unsigned __shfl_up_sync(unsigned mask, unsigned var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 136
{ } 
#endif
#line 137 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_30_intrinsics.h"
static __inline int __shfl_down_sync(unsigned mask, int var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 137
{ } 
#endif
#line 138 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_30_intrinsics.h"
static __inline unsigned __shfl_down_sync(unsigned mask, unsigned var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 138
{ } 
#endif
#line 139 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_30_intrinsics.h"
static __inline int __shfl_xor_sync(unsigned mask, int var, int laneMask, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
#line 139
{ } 
#endif
#line 140 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_30_intrinsics.h"
static __inline unsigned __shfl_xor_sync(unsigned mask, unsigned var, int laneMask, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
#line 140
{ } 
#endif
#line 141 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_30_intrinsics.h"
static __inline float __shfl_sync(unsigned mask, float var, int srcLane, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
#line 141
{ } 
#endif
#line 142 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_30_intrinsics.h"
static __inline float __shfl_up_sync(unsigned mask, float var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 142
{ } 
#endif
#line 143 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_30_intrinsics.h"
static __inline float __shfl_down_sync(unsigned mask, float var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 143
{ } 
#endif
#line 144 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_30_intrinsics.h"
static __inline float __shfl_xor_sync(unsigned mask, float var, int laneMask, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
#line 144
{ } 
#endif
#line 148 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_30_intrinsics.h"
__declspec(deprecated("__shfl() is deprecated in favor of __shfl_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to sup" "press this warning).")) static __inline unsigned __int64 __shfl(unsigned __int64 var, int srcLane, int width = 32) {int volatile ___ = 1;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
#line 148
{ } 
#endif
#line 149 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_30_intrinsics.h"
__declspec(deprecated("__shfl() is deprecated in favor of __shfl_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to sup" "press this warning).")) static __inline __int64 __shfl(__int64 var, int srcLane, int width = 32) {int volatile ___ = 1;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
#line 149
{ } 
#endif
#line 150 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_30_intrinsics.h"
__declspec(deprecated("__shfl_up() is deprecated in favor of __shfl_up_sync() and may be removed in a future release (Use -Wno-deprecated-declarations " "to suppress this warning).")) static __inline __int64 __shfl_up(__int64 var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 150
{ } 
#endif
#line 151 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_30_intrinsics.h"
__declspec(deprecated("__shfl_up() is deprecated in favor of __shfl_up_sync() and may be removed in a future release (Use -Wno-deprecated-declarations " "to suppress this warning).")) static __inline unsigned __int64 __shfl_up(unsigned __int64 var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 151
{ } 
#endif
#line 152 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_30_intrinsics.h"
__declspec(deprecated("__shfl_down() is deprecated in favor of __shfl_down_sync() and may be removed in a future release (Use -Wno-deprecated-declarati" "ons to suppress this warning).")) static __inline __int64 __shfl_down(__int64 var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 152
{ } 
#endif
#line 153 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_30_intrinsics.h"
__declspec(deprecated("__shfl_down() is deprecated in favor of __shfl_down_sync() and may be removed in a future release (Use -Wno-deprecated-declarati" "ons to suppress this warning).")) static __inline unsigned __int64 __shfl_down(unsigned __int64 var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 153
{ } 
#endif
#line 154 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_30_intrinsics.h"
__declspec(deprecated("__shfl_xor() is deprecated in favor of __shfl_xor_sync() and may be removed in a future release (Use -Wno-deprecated-declaration" "s to suppress this warning).")) static __inline __int64 __shfl_xor(__int64 var, int laneMask, int width = 32) {int volatile ___ = 1;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
#line 154
{ } 
#endif
#line 155 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_30_intrinsics.h"
__declspec(deprecated("__shfl_xor() is deprecated in favor of __shfl_xor_sync() and may be removed in a future release (Use -Wno-deprecated-declaration" "s to suppress this warning).")) static __inline unsigned __int64 __shfl_xor(unsigned __int64 var, int laneMask, int width = 32) {int volatile ___ = 1;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
#line 155
{ } 
#endif
#line 156 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_30_intrinsics.h"
__declspec(deprecated("__shfl() is deprecated in favor of __shfl_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to sup" "press this warning).")) static __inline double __shfl(double var, int srcLane, int width = 32) {int volatile ___ = 1;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
#line 156
{ } 
#endif
#line 157 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_30_intrinsics.h"
__declspec(deprecated("__shfl_up() is deprecated in favor of __shfl_up_sync() and may be removed in a future release (Use -Wno-deprecated-declarations " "to suppress this warning).")) static __inline double __shfl_up(double var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 157
{ } 
#endif
#line 158 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_30_intrinsics.h"
__declspec(deprecated("__shfl_down() is deprecated in favor of __shfl_down_sync() and may be removed in a future release (Use -Wno-deprecated-declarati" "ons to suppress this warning).")) static __inline double __shfl_down(double var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 158
{ } 
#endif
#line 159 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_30_intrinsics.h"
__declspec(deprecated("__shfl_xor() is deprecated in favor of __shfl_xor_sync() and may be removed in a future release (Use -Wno-deprecated-declaration" "s to suppress this warning).")) static __inline double __shfl_xor(double var, int laneMask, int width = 32) {int volatile ___ = 1;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
#line 159
{ } 
#endif
#line 162 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_30_intrinsics.h"
static __inline __int64 __shfl_sync(unsigned mask, __int64 var, int srcLane, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
#line 162
{ } 
#endif
#line 163 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_30_intrinsics.h"
static __inline unsigned __int64 __shfl_sync(unsigned mask, unsigned __int64 var, int srcLane, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
#line 163
{ } 
#endif
#line 164 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_30_intrinsics.h"
static __inline __int64 __shfl_up_sync(unsigned mask, __int64 var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 164
{ } 
#endif
#line 165 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_30_intrinsics.h"
static __inline unsigned __int64 __shfl_up_sync(unsigned mask, unsigned __int64 var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 165
{ } 
#endif
#line 166 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_30_intrinsics.h"
static __inline __int64 __shfl_down_sync(unsigned mask, __int64 var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 166
{ } 
#endif
#line 167 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_30_intrinsics.h"
static __inline unsigned __int64 __shfl_down_sync(unsigned mask, unsigned __int64 var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 167
{ } 
#endif
#line 168 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_30_intrinsics.h"
static __inline __int64 __shfl_xor_sync(unsigned mask, __int64 var, int laneMask, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
#line 168
{ } 
#endif
#line 169 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_30_intrinsics.h"
static __inline unsigned __int64 __shfl_xor_sync(unsigned mask, unsigned __int64 var, int laneMask, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
#line 169
{ } 
#endif
#line 170 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_30_intrinsics.h"
static __inline double __shfl_sync(unsigned mask, double var, int srcLane, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
#line 170
{ } 
#endif
#line 171 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_30_intrinsics.h"
static __inline double __shfl_up_sync(unsigned mask, double var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 171
{ } 
#endif
#line 172 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_30_intrinsics.h"
static __inline double __shfl_down_sync(unsigned mask, double var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 172
{ } 
#endif
#line 173 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_30_intrinsics.h"
static __inline double __shfl_xor_sync(unsigned mask, double var, int laneMask, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
#line 173
{ } 
#endif
#line 177 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_30_intrinsics.h"
__declspec(deprecated("__shfl() is deprecated in favor of __shfl_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to sup" "press this warning).")) static __inline long __shfl(long var, int srcLane, int width = 32) {int volatile ___ = 1;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
#line 177
{ } 
#endif
#line 178 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_30_intrinsics.h"
__declspec(deprecated("__shfl() is deprecated in favor of __shfl_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to sup" "press this warning).")) static __inline unsigned long __shfl(unsigned long var, int srcLane, int width = 32) {int volatile ___ = 1;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
#line 178
{ } 
#endif
#line 179 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_30_intrinsics.h"
__declspec(deprecated("__shfl_up() is deprecated in favor of __shfl_up_sync() and may be removed in a future release (Use -Wno-deprecated-declarations " "to suppress this warning).")) static __inline long __shfl_up(long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 179
{ } 
#endif
#line 180 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_30_intrinsics.h"
__declspec(deprecated("__shfl_up() is deprecated in favor of __shfl_up_sync() and may be removed in a future release (Use -Wno-deprecated-declarations " "to suppress this warning).")) static __inline unsigned long __shfl_up(unsigned long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 180
{ } 
#endif
#line 181 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_30_intrinsics.h"
__declspec(deprecated("__shfl_down() is deprecated in favor of __shfl_down_sync() and may be removed in a future release (Use -Wno-deprecated-declarati" "ons to suppress this warning).")) static __inline long __shfl_down(long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 181
{ } 
#endif
#line 182 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_30_intrinsics.h"
__declspec(deprecated("__shfl_down() is deprecated in favor of __shfl_down_sync() and may be removed in a future release (Use -Wno-deprecated-declarati" "ons to suppress this warning).")) static __inline unsigned long __shfl_down(unsigned long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 182
{ } 
#endif
#line 183 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_30_intrinsics.h"
__declspec(deprecated("__shfl_xor() is deprecated in favor of __shfl_xor_sync() and may be removed in a future release (Use -Wno-deprecated-declaration" "s to suppress this warning).")) static __inline long __shfl_xor(long var, int laneMask, int width = 32) {int volatile ___ = 1;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
#line 183
{ } 
#endif
#line 184 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_30_intrinsics.h"
__declspec(deprecated("__shfl_xor() is deprecated in favor of __shfl_xor_sync() and may be removed in a future release (Use -Wno-deprecated-declaration" "s to suppress this warning).")) static __inline unsigned long __shfl_xor(unsigned long var, int laneMask, int width = 32) {int volatile ___ = 1;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
#line 184
{ } 
#endif
#line 187 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_30_intrinsics.h"
static __inline long __shfl_sync(unsigned mask, long var, int srcLane, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
#line 187
{ } 
#endif
#line 188 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_30_intrinsics.h"
static __inline unsigned long __shfl_sync(unsigned mask, unsigned long var, int srcLane, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
#line 188
{ } 
#endif
#line 189 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_30_intrinsics.h"
static __inline long __shfl_up_sync(unsigned mask, long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 189
{ } 
#endif
#line 190 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_30_intrinsics.h"
static __inline unsigned long __shfl_up_sync(unsigned mask, unsigned long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 190
{ } 
#endif
#line 191 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_30_intrinsics.h"
static __inline long __shfl_down_sync(unsigned mask, long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 191
{ } 
#endif
#line 192 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_30_intrinsics.h"
static __inline unsigned long __shfl_down_sync(unsigned mask, unsigned long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 192
{ } 
#endif
#line 193 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_30_intrinsics.h"
static __inline long __shfl_xor_sync(unsigned mask, long var, int laneMask, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
#line 193
{ } 
#endif
#line 194 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_30_intrinsics.h"
static __inline unsigned long __shfl_xor_sync(unsigned mask, unsigned long var, int laneMask, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
#line 194
{ } 
#endif
#line 87 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline long __ldg(const long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 87
{ } 
#endif
#line 88 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline unsigned long __ldg(const unsigned long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 88
{ } 
#endif
#line 90 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline char __ldg(const char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 90
{ } 
#endif
#line 91 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline signed char __ldg(const signed char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 91
{ } 
#endif
#line 92 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline short __ldg(const short *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 92
{ } 
#endif
#line 93 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline int __ldg(const int *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 93
{ } 
#endif
#line 94 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline __int64 __ldg(const __int64 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 94
{ } 
#endif
#line 95 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline char2 __ldg(const char2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 95
{ } 
#endif
#line 96 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline char4 __ldg(const char4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 96
{ } 
#endif
#line 97 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline short2 __ldg(const short2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 97
{ } 
#endif
#line 98 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline short4 __ldg(const short4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 98
{ } 
#endif
#line 99 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline int2 __ldg(const int2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 99
{ } 
#endif
#line 100 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline int4 __ldg(const int4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 100
{ } 
#endif
#line 101 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline longlong2 __ldg(const longlong2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 101
{ } 
#endif
#line 103 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline unsigned char __ldg(const unsigned char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 103
{ } 
#endif
#line 104 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline unsigned short __ldg(const unsigned short *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 104
{ } 
#endif
#line 105 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline unsigned __ldg(const unsigned *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 105
{ } 
#endif
#line 106 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline unsigned __int64 __ldg(const unsigned __int64 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 106
{ } 
#endif
#line 107 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline uchar2 __ldg(const uchar2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 107
{ } 
#endif
#line 108 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline uchar4 __ldg(const uchar4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 108
{ } 
#endif
#line 109 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline ushort2 __ldg(const ushort2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 109
{ } 
#endif
#line 110 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline ushort4 __ldg(const ushort4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 110
{ } 
#endif
#line 111 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline uint2 __ldg(const uint2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 111
{ } 
#endif
#line 112 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline uint4 __ldg(const uint4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 112
{ } 
#endif
#line 113 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline ulonglong2 __ldg(const ulonglong2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 113
{ } 
#endif
#line 115 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline float __ldg(const float *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 115
{ } 
#endif
#line 116 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline double __ldg(const double *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 116
{ } 
#endif
#line 117 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline float2 __ldg(const float2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 117
{ } 
#endif
#line 118 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline float4 __ldg(const float4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 118
{ } 
#endif
#line 119 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline double2 __ldg(const double2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 119
{ } 
#endif
#line 123 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline long __ldcg(const long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 123
{ } 
#endif
#line 124 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline unsigned long __ldcg(const unsigned long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 124
{ } 
#endif
#line 126 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline char __ldcg(const char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 126
{ } 
#endif
#line 127 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline signed char __ldcg(const signed char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 127
{ } 
#endif
#line 128 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline short __ldcg(const short *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 128
{ } 
#endif
#line 129 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline int __ldcg(const int *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 129
{ } 
#endif
#line 130 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline __int64 __ldcg(const __int64 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 130
{ } 
#endif
#line 131 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline char2 __ldcg(const char2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 131
{ } 
#endif
#line 132 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline char4 __ldcg(const char4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 132
{ } 
#endif
#line 133 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline short2 __ldcg(const short2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 133
{ } 
#endif
#line 134 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline short4 __ldcg(const short4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 134
{ } 
#endif
#line 135 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline int2 __ldcg(const int2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 135
{ } 
#endif
#line 136 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline int4 __ldcg(const int4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 136
{ } 
#endif
#line 137 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline longlong2 __ldcg(const longlong2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 137
{ } 
#endif
#line 139 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline unsigned char __ldcg(const unsigned char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 139
{ } 
#endif
#line 140 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline unsigned short __ldcg(const unsigned short *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 140
{ } 
#endif
#line 141 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline unsigned __ldcg(const unsigned *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 141
{ } 
#endif
#line 142 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline unsigned __int64 __ldcg(const unsigned __int64 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 142
{ } 
#endif
#line 143 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline uchar2 __ldcg(const uchar2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 143
{ } 
#endif
#line 144 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline uchar4 __ldcg(const uchar4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 144
{ } 
#endif
#line 145 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline ushort2 __ldcg(const ushort2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 145
{ } 
#endif
#line 146 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline ushort4 __ldcg(const ushort4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 146
{ } 
#endif
#line 147 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline uint2 __ldcg(const uint2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 147
{ } 
#endif
#line 148 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline uint4 __ldcg(const uint4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 148
{ } 
#endif
#line 149 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline ulonglong2 __ldcg(const ulonglong2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 149
{ } 
#endif
#line 151 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline float __ldcg(const float *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 151
{ } 
#endif
#line 152 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline double __ldcg(const double *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 152
{ } 
#endif
#line 153 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline float2 __ldcg(const float2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 153
{ } 
#endif
#line 154 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline float4 __ldcg(const float4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 154
{ } 
#endif
#line 155 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline double2 __ldcg(const double2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 155
{ } 
#endif
#line 159 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline long __ldca(const long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 159
{ } 
#endif
#line 160 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline unsigned long __ldca(const unsigned long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 160
{ } 
#endif
#line 162 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline char __ldca(const char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 162
{ } 
#endif
#line 163 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline signed char __ldca(const signed char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 163
{ } 
#endif
#line 164 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline short __ldca(const short *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 164
{ } 
#endif
#line 165 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline int __ldca(const int *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 165
{ } 
#endif
#line 166 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline __int64 __ldca(const __int64 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 166
{ } 
#endif
#line 167 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline char2 __ldca(const char2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 167
{ } 
#endif
#line 168 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline char4 __ldca(const char4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 168
{ } 
#endif
#line 169 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline short2 __ldca(const short2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 169
{ } 
#endif
#line 170 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline short4 __ldca(const short4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 170
{ } 
#endif
#line 171 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline int2 __ldca(const int2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 171
{ } 
#endif
#line 172 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline int4 __ldca(const int4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 172
{ } 
#endif
#line 173 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline longlong2 __ldca(const longlong2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 173
{ } 
#endif
#line 175 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline unsigned char __ldca(const unsigned char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 175
{ } 
#endif
#line 176 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline unsigned short __ldca(const unsigned short *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 176
{ } 
#endif
#line 177 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline unsigned __ldca(const unsigned *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 177
{ } 
#endif
#line 178 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline unsigned __int64 __ldca(const unsigned __int64 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 178
{ } 
#endif
#line 179 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline uchar2 __ldca(const uchar2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 179
{ } 
#endif
#line 180 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline uchar4 __ldca(const uchar4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 180
{ } 
#endif
#line 181 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline ushort2 __ldca(const ushort2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 181
{ } 
#endif
#line 182 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline ushort4 __ldca(const ushort4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 182
{ } 
#endif
#line 183 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline uint2 __ldca(const uint2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 183
{ } 
#endif
#line 184 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline uint4 __ldca(const uint4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 184
{ } 
#endif
#line 185 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline ulonglong2 __ldca(const ulonglong2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 185
{ } 
#endif
#line 187 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline float __ldca(const float *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 187
{ } 
#endif
#line 188 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline double __ldca(const double *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 188
{ } 
#endif
#line 189 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline float2 __ldca(const float2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 189
{ } 
#endif
#line 190 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline float4 __ldca(const float4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 190
{ } 
#endif
#line 191 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline double2 __ldca(const double2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 191
{ } 
#endif
#line 195 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline long __ldcs(const long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 195
{ } 
#endif
#line 196 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline unsigned long __ldcs(const unsigned long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 196
{ } 
#endif
#line 198 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline char __ldcs(const char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 198
{ } 
#endif
#line 199 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline signed char __ldcs(const signed char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 199
{ } 
#endif
#line 200 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline short __ldcs(const short *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 200
{ } 
#endif
#line 201 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline int __ldcs(const int *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 201
{ } 
#endif
#line 202 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline __int64 __ldcs(const __int64 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 202
{ } 
#endif
#line 203 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline char2 __ldcs(const char2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 203
{ } 
#endif
#line 204 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline char4 __ldcs(const char4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 204
{ } 
#endif
#line 205 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline short2 __ldcs(const short2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 205
{ } 
#endif
#line 206 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline short4 __ldcs(const short4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 206
{ } 
#endif
#line 207 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline int2 __ldcs(const int2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 207
{ } 
#endif
#line 208 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline int4 __ldcs(const int4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 208
{ } 
#endif
#line 209 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline longlong2 __ldcs(const longlong2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 209
{ } 
#endif
#line 211 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline unsigned char __ldcs(const unsigned char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 211
{ } 
#endif
#line 212 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline unsigned short __ldcs(const unsigned short *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 212
{ } 
#endif
#line 213 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline unsigned __ldcs(const unsigned *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 213
{ } 
#endif
#line 214 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline unsigned __int64 __ldcs(const unsigned __int64 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 214
{ } 
#endif
#line 215 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline uchar2 __ldcs(const uchar2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 215
{ } 
#endif
#line 216 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline uchar4 __ldcs(const uchar4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 216
{ } 
#endif
#line 217 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline ushort2 __ldcs(const ushort2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 217
{ } 
#endif
#line 218 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline ushort4 __ldcs(const ushort4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 218
{ } 
#endif
#line 219 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline uint2 __ldcs(const uint2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 219
{ } 
#endif
#line 220 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline uint4 __ldcs(const uint4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 220
{ } 
#endif
#line 221 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline ulonglong2 __ldcs(const ulonglong2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 221
{ } 
#endif
#line 223 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline float __ldcs(const float *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 223
{ } 
#endif
#line 224 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline double __ldcs(const double *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 224
{ } 
#endif
#line 225 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline float2 __ldcs(const float2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 225
{ } 
#endif
#line 226 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline float4 __ldcs(const float4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 226
{ } 
#endif
#line 227 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline double2 __ldcs(const double2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 227
{ } 
#endif
#line 231 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline long __ldlu(const long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 231
{ } 
#endif
#line 232 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline unsigned long __ldlu(const unsigned long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 232
{ } 
#endif
#line 234 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline char __ldlu(const char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 234
{ } 
#endif
#line 235 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline signed char __ldlu(const signed char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 235
{ } 
#endif
#line 236 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline short __ldlu(const short *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 236
{ } 
#endif
#line 237 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline int __ldlu(const int *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 237
{ } 
#endif
#line 238 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline __int64 __ldlu(const __int64 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 238
{ } 
#endif
#line 239 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline char2 __ldlu(const char2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 239
{ } 
#endif
#line 240 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline char4 __ldlu(const char4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 240
{ } 
#endif
#line 241 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline short2 __ldlu(const short2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 241
{ } 
#endif
#line 242 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline short4 __ldlu(const short4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 242
{ } 
#endif
#line 243 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline int2 __ldlu(const int2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 243
{ } 
#endif
#line 244 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline int4 __ldlu(const int4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 244
{ } 
#endif
#line 245 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline longlong2 __ldlu(const longlong2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 245
{ } 
#endif
#line 247 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline unsigned char __ldlu(const unsigned char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 247
{ } 
#endif
#line 248 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline unsigned short __ldlu(const unsigned short *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 248
{ } 
#endif
#line 249 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline unsigned __ldlu(const unsigned *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 249
{ } 
#endif
#line 250 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline unsigned __int64 __ldlu(const unsigned __int64 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 250
{ } 
#endif
#line 251 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline uchar2 __ldlu(const uchar2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 251
{ } 
#endif
#line 252 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline uchar4 __ldlu(const uchar4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 252
{ } 
#endif
#line 253 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline ushort2 __ldlu(const ushort2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 253
{ } 
#endif
#line 254 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline ushort4 __ldlu(const ushort4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 254
{ } 
#endif
#line 255 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline uint2 __ldlu(const uint2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 255
{ } 
#endif
#line 256 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline uint4 __ldlu(const uint4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 256
{ } 
#endif
#line 257 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline ulonglong2 __ldlu(const ulonglong2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 257
{ } 
#endif
#line 259 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline float __ldlu(const float *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 259
{ } 
#endif
#line 260 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline double __ldlu(const double *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 260
{ } 
#endif
#line 261 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline float2 __ldlu(const float2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 261
{ } 
#endif
#line 262 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline float4 __ldlu(const float4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 262
{ } 
#endif
#line 263 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline double2 __ldlu(const double2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 263
{ } 
#endif
#line 267 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline long __ldcv(const long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 267
{ } 
#endif
#line 268 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline unsigned long __ldcv(const unsigned long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 268
{ } 
#endif
#line 270 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline char __ldcv(const char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 270
{ } 
#endif
#line 271 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline signed char __ldcv(const signed char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 271
{ } 
#endif
#line 272 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline short __ldcv(const short *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 272
{ } 
#endif
#line 273 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline int __ldcv(const int *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 273
{ } 
#endif
#line 274 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline __int64 __ldcv(const __int64 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 274
{ } 
#endif
#line 275 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline char2 __ldcv(const char2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 275
{ } 
#endif
#line 276 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline char4 __ldcv(const char4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 276
{ } 
#endif
#line 277 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline short2 __ldcv(const short2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 277
{ } 
#endif
#line 278 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline short4 __ldcv(const short4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 278
{ } 
#endif
#line 279 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline int2 __ldcv(const int2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 279
{ } 
#endif
#line 280 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline int4 __ldcv(const int4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 280
{ } 
#endif
#line 281 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline longlong2 __ldcv(const longlong2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 281
{ } 
#endif
#line 283 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline unsigned char __ldcv(const unsigned char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 283
{ } 
#endif
#line 284 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline unsigned short __ldcv(const unsigned short *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 284
{ } 
#endif
#line 285 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline unsigned __ldcv(const unsigned *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 285
{ } 
#endif
#line 286 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline unsigned __int64 __ldcv(const unsigned __int64 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 286
{ } 
#endif
#line 287 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline uchar2 __ldcv(const uchar2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 287
{ } 
#endif
#line 288 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline uchar4 __ldcv(const uchar4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 288
{ } 
#endif
#line 289 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline ushort2 __ldcv(const ushort2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 289
{ } 
#endif
#line 290 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline ushort4 __ldcv(const ushort4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 290
{ } 
#endif
#line 291 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline uint2 __ldcv(const uint2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 291
{ } 
#endif
#line 292 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline uint4 __ldcv(const uint4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 292
{ } 
#endif
#line 293 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline ulonglong2 __ldcv(const ulonglong2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 293
{ } 
#endif
#line 295 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline float __ldcv(const float *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 295
{ } 
#endif
#line 296 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline double __ldcv(const double *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 296
{ } 
#endif
#line 297 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline float2 __ldcv(const float2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 297
{ } 
#endif
#line 298 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline float4 __ldcv(const float4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 298
{ } 
#endif
#line 299 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline double2 __ldcv(const double2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 299
{ } 
#endif
#line 303 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline void __stwb(long *ptr, long value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 303
{ } 
#endif
#line 304 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline void __stwb(unsigned long *ptr, unsigned long value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 304
{ } 
#endif
#line 306 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline void __stwb(char *ptr, char value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 306
{ } 
#endif
#line 307 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline void __stwb(signed char *ptr, signed char value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 307
{ } 
#endif
#line 308 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline void __stwb(short *ptr, short value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 308
{ } 
#endif
#line 309 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline void __stwb(int *ptr, int value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 309
{ } 
#endif
#line 310 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline void __stwb(__int64 *ptr, __int64 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 310
{ } 
#endif
#line 311 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline void __stwb(char2 *ptr, char2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 311
{ } 
#endif
#line 312 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline void __stwb(char4 *ptr, char4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 312
{ } 
#endif
#line 313 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline void __stwb(short2 *ptr, short2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 313
{ } 
#endif
#line 314 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline void __stwb(short4 *ptr, short4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 314
{ } 
#endif
#line 315 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline void __stwb(int2 *ptr, int2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 315
{ } 
#endif
#line 316 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline void __stwb(int4 *ptr, int4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 316
{ } 
#endif
#line 317 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline void __stwb(longlong2 *ptr, longlong2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 317
{ } 
#endif
#line 319 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline void __stwb(unsigned char *ptr, unsigned char value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 319
{ } 
#endif
#line 320 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline void __stwb(unsigned short *ptr, unsigned short value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 320
{ } 
#endif
#line 321 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline void __stwb(unsigned *ptr, unsigned value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 321
{ } 
#endif
#line 322 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline void __stwb(unsigned __int64 *ptr, unsigned __int64 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 322
{ } 
#endif
#line 323 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline void __stwb(uchar2 *ptr, uchar2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 323
{ } 
#endif
#line 324 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline void __stwb(uchar4 *ptr, uchar4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 324
{ } 
#endif
#line 325 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline void __stwb(ushort2 *ptr, ushort2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 325
{ } 
#endif
#line 326 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline void __stwb(ushort4 *ptr, ushort4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 326
{ } 
#endif
#line 327 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline void __stwb(uint2 *ptr, uint2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 327
{ } 
#endif
#line 328 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline void __stwb(uint4 *ptr, uint4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 328
{ } 
#endif
#line 329 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline void __stwb(ulonglong2 *ptr, ulonglong2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 329
{ } 
#endif
#line 331 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline void __stwb(float *ptr, float value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 331
{ } 
#endif
#line 332 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline void __stwb(double *ptr, double value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 332
{ } 
#endif
#line 333 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline void __stwb(float2 *ptr, float2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 333
{ } 
#endif
#line 334 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline void __stwb(float4 *ptr, float4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 334
{ } 
#endif
#line 335 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline void __stwb(double2 *ptr, double2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 335
{ } 
#endif
#line 339 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline void __stcg(long *ptr, long value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 339
{ } 
#endif
#line 340 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline void __stcg(unsigned long *ptr, unsigned long value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 340
{ } 
#endif
#line 342 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline void __stcg(char *ptr, char value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 342
{ } 
#endif
#line 343 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline void __stcg(signed char *ptr, signed char value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 343
{ } 
#endif
#line 344 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline void __stcg(short *ptr, short value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 344
{ } 
#endif
#line 345 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline void __stcg(int *ptr, int value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 345
{ } 
#endif
#line 346 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline void __stcg(__int64 *ptr, __int64 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 346
{ } 
#endif
#line 347 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline void __stcg(char2 *ptr, char2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 347
{ } 
#endif
#line 348 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline void __stcg(char4 *ptr, char4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 348
{ } 
#endif
#line 349 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline void __stcg(short2 *ptr, short2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 349
{ } 
#endif
#line 350 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline void __stcg(short4 *ptr, short4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 350
{ } 
#endif
#line 351 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline void __stcg(int2 *ptr, int2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 351
{ } 
#endif
#line 352 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline void __stcg(int4 *ptr, int4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 352
{ } 
#endif
#line 353 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline void __stcg(longlong2 *ptr, longlong2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 353
{ } 
#endif
#line 355 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline void __stcg(unsigned char *ptr, unsigned char value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 355
{ } 
#endif
#line 356 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline void __stcg(unsigned short *ptr, unsigned short value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 356
{ } 
#endif
#line 357 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline void __stcg(unsigned *ptr, unsigned value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 357
{ } 
#endif
#line 358 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline void __stcg(unsigned __int64 *ptr, unsigned __int64 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 358
{ } 
#endif
#line 359 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline void __stcg(uchar2 *ptr, uchar2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 359
{ } 
#endif
#line 360 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline void __stcg(uchar4 *ptr, uchar4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 360
{ } 
#endif
#line 361 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline void __stcg(ushort2 *ptr, ushort2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 361
{ } 
#endif
#line 362 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline void __stcg(ushort4 *ptr, ushort4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 362
{ } 
#endif
#line 363 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline void __stcg(uint2 *ptr, uint2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 363
{ } 
#endif
#line 364 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline void __stcg(uint4 *ptr, uint4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 364
{ } 
#endif
#line 365 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline void __stcg(ulonglong2 *ptr, ulonglong2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 365
{ } 
#endif
#line 367 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline void __stcg(float *ptr, float value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 367
{ } 
#endif
#line 368 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline void __stcg(double *ptr, double value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 368
{ } 
#endif
#line 369 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline void __stcg(float2 *ptr, float2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 369
{ } 
#endif
#line 370 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline void __stcg(float4 *ptr, float4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 370
{ } 
#endif
#line 371 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline void __stcg(double2 *ptr, double2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 371
{ } 
#endif
#line 375 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline void __stcs(long *ptr, long value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 375
{ } 
#endif
#line 376 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline void __stcs(unsigned long *ptr, unsigned long value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 376
{ } 
#endif
#line 378 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline void __stcs(char *ptr, char value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 378
{ } 
#endif
#line 379 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline void __stcs(signed char *ptr, signed char value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 379
{ } 
#endif
#line 380 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline void __stcs(short *ptr, short value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 380
{ } 
#endif
#line 381 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline void __stcs(int *ptr, int value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 381
{ } 
#endif
#line 382 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline void __stcs(__int64 *ptr, __int64 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 382
{ } 
#endif
#line 383 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline void __stcs(char2 *ptr, char2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 383
{ } 
#endif
#line 384 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline void __stcs(char4 *ptr, char4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 384
{ } 
#endif
#line 385 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline void __stcs(short2 *ptr, short2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 385
{ } 
#endif
#line 386 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline void __stcs(short4 *ptr, short4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 386
{ } 
#endif
#line 387 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline void __stcs(int2 *ptr, int2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 387
{ } 
#endif
#line 388 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline void __stcs(int4 *ptr, int4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 388
{ } 
#endif
#line 389 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline void __stcs(longlong2 *ptr, longlong2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 389
{ } 
#endif
#line 391 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline void __stcs(unsigned char *ptr, unsigned char value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 391
{ } 
#endif
#line 392 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline void __stcs(unsigned short *ptr, unsigned short value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 392
{ } 
#endif
#line 393 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline void __stcs(unsigned *ptr, unsigned value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 393
{ } 
#endif
#line 394 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline void __stcs(unsigned __int64 *ptr, unsigned __int64 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 394
{ } 
#endif
#line 395 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline void __stcs(uchar2 *ptr, uchar2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 395
{ } 
#endif
#line 396 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline void __stcs(uchar4 *ptr, uchar4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 396
{ } 
#endif
#line 397 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline void __stcs(ushort2 *ptr, ushort2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 397
{ } 
#endif
#line 398 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline void __stcs(ushort4 *ptr, ushort4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 398
{ } 
#endif
#line 399 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline void __stcs(uint2 *ptr, uint2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 399
{ } 
#endif
#line 400 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline void __stcs(uint4 *ptr, uint4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 400
{ } 
#endif
#line 401 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline void __stcs(ulonglong2 *ptr, ulonglong2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 401
{ } 
#endif
#line 403 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline void __stcs(float *ptr, float value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 403
{ } 
#endif
#line 404 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline void __stcs(double *ptr, double value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 404
{ } 
#endif
#line 405 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline void __stcs(float2 *ptr, float2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 405
{ } 
#endif
#line 406 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline void __stcs(float4 *ptr, float4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 406
{ } 
#endif
#line 407 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline void __stcs(double2 *ptr, double2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 407
{ } 
#endif
#line 411 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline void __stwt(long *ptr, long value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 411
{ } 
#endif
#line 412 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline void __stwt(unsigned long *ptr, unsigned long value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 412
{ } 
#endif
#line 414 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline void __stwt(char *ptr, char value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 414
{ } 
#endif
#line 415 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline void __stwt(signed char *ptr, signed char value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 415
{ } 
#endif
#line 416 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline void __stwt(short *ptr, short value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 416
{ } 
#endif
#line 417 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline void __stwt(int *ptr, int value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 417
{ } 
#endif
#line 418 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline void __stwt(__int64 *ptr, __int64 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 418
{ } 
#endif
#line 419 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline void __stwt(char2 *ptr, char2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 419
{ } 
#endif
#line 420 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline void __stwt(char4 *ptr, char4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 420
{ } 
#endif
#line 421 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline void __stwt(short2 *ptr, short2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 421
{ } 
#endif
#line 422 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline void __stwt(short4 *ptr, short4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 422
{ } 
#endif
#line 423 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline void __stwt(int2 *ptr, int2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 423
{ } 
#endif
#line 424 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline void __stwt(int4 *ptr, int4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 424
{ } 
#endif
#line 425 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline void __stwt(longlong2 *ptr, longlong2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 425
{ } 
#endif
#line 427 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline void __stwt(unsigned char *ptr, unsigned char value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 427
{ } 
#endif
#line 428 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline void __stwt(unsigned short *ptr, unsigned short value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 428
{ } 
#endif
#line 429 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline void __stwt(unsigned *ptr, unsigned value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 429
{ } 
#endif
#line 430 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline void __stwt(unsigned __int64 *ptr, unsigned __int64 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 430
{ } 
#endif
#line 431 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline void __stwt(uchar2 *ptr, uchar2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 431
{ } 
#endif
#line 432 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline void __stwt(uchar4 *ptr, uchar4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 432
{ } 
#endif
#line 433 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline void __stwt(ushort2 *ptr, ushort2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 433
{ } 
#endif
#line 434 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline void __stwt(ushort4 *ptr, ushort4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 434
{ } 
#endif
#line 435 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline void __stwt(uint2 *ptr, uint2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 435
{ } 
#endif
#line 436 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline void __stwt(uint4 *ptr, uint4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 436
{ } 
#endif
#line 437 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline void __stwt(ulonglong2 *ptr, ulonglong2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 437
{ } 
#endif
#line 439 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline void __stwt(float *ptr, float value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 439
{ } 
#endif
#line 440 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline void __stwt(double *ptr, double value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 440
{ } 
#endif
#line 441 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline void __stwt(float2 *ptr, float2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 441
{ } 
#endif
#line 442 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline void __stwt(float4 *ptr, float4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 442
{ } 
#endif
#line 443 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline void __stwt(double2 *ptr, double2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 443
{ } 
#endif
#line 460 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline unsigned __funnelshift_l(unsigned lo, unsigned hi, unsigned shift) {int volatile ___ = 1;(void)lo;(void)hi;(void)shift;::exit(___);}
#if 0
#line 460
{ } 
#endif
#line 472 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline unsigned __funnelshift_lc(unsigned lo, unsigned hi, unsigned shift) {int volatile ___ = 1;(void)lo;(void)hi;(void)shift;::exit(___);}
#if 0
#line 472
{ } 
#endif
#line 485 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline unsigned __funnelshift_r(unsigned lo, unsigned hi, unsigned shift) {int volatile ___ = 1;(void)lo;(void)hi;(void)shift;::exit(___);}
#if 0
#line 485
{ } 
#endif
#line 497 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_32_intrinsics.h"
static __inline unsigned __funnelshift_rc(unsigned lo, unsigned hi, unsigned shift) {int volatile ___ = 1;(void)lo;(void)hi;(void)shift;::exit(___);}
#if 0
#line 497
{ } 
#endif
#line 89 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_61_intrinsics.h"
static __inline int __dp2a_lo(int srcA, int srcB, int c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
#line 89
{ } 
#endif
#line 90 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_61_intrinsics.h"
static __inline unsigned __dp2a_lo(unsigned srcA, unsigned srcB, unsigned c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
#line 90
{ } 
#endif
#line 92 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_61_intrinsics.h"
static __inline int __dp2a_lo(short2 srcA, char4 srcB, int c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
#line 92
{ } 
#endif
#line 93 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_61_intrinsics.h"
static __inline unsigned __dp2a_lo(ushort2 srcA, uchar4 srcB, unsigned c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
#line 93
{ } 
#endif
#line 95 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_61_intrinsics.h"
static __inline int __dp2a_hi(int srcA, int srcB, int c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
#line 95
{ } 
#endif
#line 96 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_61_intrinsics.h"
static __inline unsigned __dp2a_hi(unsigned srcA, unsigned srcB, unsigned c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
#line 96
{ } 
#endif
#line 98 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_61_intrinsics.h"
static __inline int __dp2a_hi(short2 srcA, char4 srcB, int c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
#line 98
{ } 
#endif
#line 99 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_61_intrinsics.h"
static __inline unsigned __dp2a_hi(ushort2 srcA, uchar4 srcB, unsigned c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
#line 99
{ } 
#endif
#line 106 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_61_intrinsics.h"
static __inline int __dp4a(int srcA, int srcB, int c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
#line 106
{ } 
#endif
#line 107 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_61_intrinsics.h"
static __inline unsigned __dp4a(unsigned srcA, unsigned srcB, unsigned c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
#line 107
{ } 
#endif
#line 109 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_61_intrinsics.h"
static __inline int __dp4a(char4 srcA, char4 srcB, int c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
#line 109
{ } 
#endif
#line 110 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\sm_61_intrinsics.h"
static __inline unsigned __dp4a(uchar4 srcA, uchar4 srcB, unsigned c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
#line 110
{ } 
#endif
#line 93 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\sm_70_rt.h"
static __inline unsigned __match_any_sync(unsigned mask, unsigned value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
#line 93
{ } 
#endif
#line 94 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\sm_70_rt.h"
static __inline unsigned __match_any_sync(unsigned mask, int value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
#line 94
{ } 
#endif
#line 95 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\sm_70_rt.h"
static __inline unsigned __match_any_sync(unsigned mask, unsigned long value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
#line 95
{ } 
#endif
#line 96 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\sm_70_rt.h"
static __inline unsigned __match_any_sync(unsigned mask, long value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
#line 96
{ } 
#endif
#line 97 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\sm_70_rt.h"
static __inline unsigned __match_any_sync(unsigned mask, unsigned __int64 value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
#line 97
{ } 
#endif
#line 98 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\sm_70_rt.h"
static __inline unsigned __match_any_sync(unsigned mask, __int64 value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
#line 98
{ } 
#endif
#line 99 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\sm_70_rt.h"
static __inline unsigned __match_any_sync(unsigned mask, float value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
#line 99
{ } 
#endif
#line 100 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\sm_70_rt.h"
static __inline unsigned __match_any_sync(unsigned mask, double value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
#line 100
{ } 
#endif
#line 102 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\sm_70_rt.h"
static __inline unsigned __match_all_sync(unsigned mask, unsigned value, int *pred) {int volatile ___ = 1;(void)mask;(void)value;(void)pred;::exit(___);}
#if 0
#line 102
{ } 
#endif
#line 103 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\sm_70_rt.h"
static __inline unsigned __match_all_sync(unsigned mask, int value, int *pred) {int volatile ___ = 1;(void)mask;(void)value;(void)pred;::exit(___);}
#if 0
#line 103
{ } 
#endif
#line 104 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\sm_70_rt.h"
static __inline unsigned __match_all_sync(unsigned mask, unsigned long value, int *pred) {int volatile ___ = 1;(void)mask;(void)value;(void)pred;::exit(___);}
#if 0
#line 104
{ } 
#endif
#line 105 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\sm_70_rt.h"
static __inline unsigned __match_all_sync(unsigned mask, long value, int *pred) {int volatile ___ = 1;(void)mask;(void)value;(void)pred;::exit(___);}
#if 0
#line 105
{ } 
#endif
#line 106 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\sm_70_rt.h"
static __inline unsigned __match_all_sync(unsigned mask, unsigned __int64 value, int *pred) {int volatile ___ = 1;(void)mask;(void)value;(void)pred;::exit(___);}
#if 0
#line 106
{ } 
#endif
#line 107 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\sm_70_rt.h"
static __inline unsigned __match_all_sync(unsigned mask, __int64 value, int *pred) {int volatile ___ = 1;(void)mask;(void)value;(void)pred;::exit(___);}
#if 0
#line 107
{ } 
#endif
#line 108 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\sm_70_rt.h"
static __inline unsigned __match_all_sync(unsigned mask, float value, int *pred) {int volatile ___ = 1;(void)mask;(void)value;(void)pred;::exit(___);}
#if 0
#line 108
{ } 
#endif
#line 109 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\sm_70_rt.h"
static __inline unsigned __match_all_sync(unsigned mask, double value, int *pred) {int volatile ___ = 1;(void)mask;(void)value;(void)pred;::exit(___);}
#if 0
#line 109
{ } 
#endif
#line 111 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\sm_70_rt.h"
static __inline void __nanosleep(unsigned ns) {int volatile ___ = 1;(void)ns;::exit(___);}
#if 0
#line 111
{ } 
#endif
#line 113 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\sm_70_rt.h"
static __inline unsigned short atomicCAS(unsigned short *address, unsigned short compare, unsigned short val) {int volatile ___ = 1;(void)address;(void)compare;(void)val;::exit(___);}
#if 0
#line 113
{ } 
#endif
#line 93 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\sm_80_rt.h"
static __inline unsigned __reduce_add_sync(unsigned mask, unsigned value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
#line 93
{ } 
#endif
#line 94 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\sm_80_rt.h"
static __inline unsigned __reduce_min_sync(unsigned mask, unsigned value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
#line 94
{ } 
#endif
#line 95 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\sm_80_rt.h"
static __inline unsigned __reduce_max_sync(unsigned mask, unsigned value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
#line 95
{ } 
#endif
#line 97 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\sm_80_rt.h"
static __inline int __reduce_add_sync(unsigned mask, int value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
#line 97
{ } 
#endif
#line 98 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\sm_80_rt.h"
static __inline int __reduce_min_sync(unsigned mask, int value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
#line 98
{ } 
#endif
#line 99 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\sm_80_rt.h"
static __inline int __reduce_max_sync(unsigned mask, int value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
#line 99
{ } 
#endif
#line 101 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\sm_80_rt.h"
static __inline unsigned __reduce_and_sync(unsigned mask, unsigned value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
#line 101
{ } 
#endif
#line 102 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\sm_80_rt.h"
static __inline unsigned __reduce_or_sync(unsigned mask, unsigned value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
#line 102
{ } 
#endif
#line 103 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\sm_80_rt.h"
static __inline unsigned __reduce_xor_sync(unsigned mask, unsigned value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
#line 103
{ } 
#endif
#line 106 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\sm_80_rt.h"
extern "C" {
#line 107
inline void *__nv_associate_access_property(const void *ptr, unsigned __int64 
#line 108
property) {int volatile ___ = 1;(void)ptr;(void)property;::exit(___);}
#if 0
#line 108
{ 
#line 109
extern void *__nv_associate_access_property_impl(const void *, unsigned __int64); 
#line 111
return __nv_associate_access_property_impl(ptr, property); 
#line 112
} 
#endif
#line 114 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\sm_80_rt.h"
inline void __nv_memcpy_async_shared_global_4(void *dst, const void *
#line 115
src, unsigned 
#line 116
src_size) {int volatile ___ = 1;(void)dst;(void)src;(void)src_size;::exit(___);}
#if 0
#line 116
{ 
#line 117
extern void __nv_memcpy_async_shared_global_4_impl(void *, const void *, unsigned); 
#line 120
__nv_memcpy_async_shared_global_4_impl(dst, src, src_size); 
#line 121
} 
#endif
#line 123 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\sm_80_rt.h"
inline void __nv_memcpy_async_shared_global_8(void *dst, const void *
#line 124
src, unsigned 
#line 125
src_size) {int volatile ___ = 1;(void)dst;(void)src;(void)src_size;::exit(___);}
#if 0
#line 125
{ 
#line 126
extern void __nv_memcpy_async_shared_global_8_impl(void *, const void *, unsigned); 
#line 129
__nv_memcpy_async_shared_global_8_impl(dst, src, src_size); 
#line 130
} 
#endif
#line 132 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\sm_80_rt.h"
inline void __nv_memcpy_async_shared_global_16(void *dst, const void *
#line 133
src, unsigned 
#line 134
src_size) {int volatile ___ = 1;(void)dst;(void)src;(void)src_size;::exit(___);}
#if 0
#line 134
{ 
#line 135
extern void __nv_memcpy_async_shared_global_16_impl(void *, const void *, unsigned); 
#line 138
__nv_memcpy_async_shared_global_16_impl(dst, src, src_size); 
#line 139
} 
#endif
#line 141 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\sm_80_rt.h"
}
#line 89 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\sm_90_rt.h"
static __inline unsigned __isCtaShared(const void *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 89
{ } 
#endif
#line 90 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\sm_90_rt.h"
static __inline unsigned __isClusterShared(const void *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 90
{ } 
#endif
#line 91 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\sm_90_rt.h"
static __inline void *__cluster_map_shared_rank(const void *ptr, unsigned target_block_rank) {int volatile ___ = 1;(void)ptr;(void)target_block_rank;::exit(___);}
#if 0
#line 91
{ } 
#endif
#line 92 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\sm_90_rt.h"
static __inline unsigned __cluster_query_shared_rank(const void *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 92
{ } 
#endif
#line 93 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\sm_90_rt.h"
static __inline uint2 __cluster_map_shared_multicast(const void *ptr, unsigned cluster_cta_mask) {int volatile ___ = 1;(void)ptr;(void)cluster_cta_mask;::exit(___);}
#if 0
#line 93
{ } 
#endif
#line 94 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\sm_90_rt.h"
static __inline unsigned __clusterDimIsSpecified() {int volatile ___ = 1;::exit(___);}
#if 0
#line 94
{ } 
#endif
#line 95 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\sm_90_rt.h"
static __inline dim3 __clusterDim() {int volatile ___ = 1;::exit(___);}
#if 0
#line 95
{ } 
#endif
#line 96 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\sm_90_rt.h"
static __inline dim3 __clusterRelativeBlockIdx() {int volatile ___ = 1;::exit(___);}
#if 0
#line 96
{ } 
#endif
#line 97 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\sm_90_rt.h"
static __inline dim3 __clusterGridDimInClusters() {int volatile ___ = 1;::exit(___);}
#if 0
#line 97
{ } 
#endif
#line 98 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\sm_90_rt.h"
static __inline dim3 __clusterIdx() {int volatile ___ = 1;::exit(___);}
#if 0
#line 98
{ } 
#endif
#line 99 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\sm_90_rt.h"
static __inline unsigned __clusterRelativeBlockRank() {int volatile ___ = 1;::exit(___);}
#if 0
#line 99
{ } 
#endif
#line 100 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\sm_90_rt.h"
static __inline unsigned __clusterSizeInBlocks() {int volatile ___ = 1;::exit(___);}
#if 0
#line 100
{ } 
#endif
#line 101 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\sm_90_rt.h"
static __inline void __cluster_barrier_arrive() {int volatile ___ = 1;::exit(___);}
#if 0
#line 101
{ } 
#endif
#line 102 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\sm_90_rt.h"
static __inline void __cluster_barrier_wait() {int volatile ___ = 1;::exit(___);}
#if 0
#line 102
{ } 
#endif
#line 103 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt\\sm_90_rt.h"
static __inline void __threadfence_cluster() {int volatile ___ = 1;::exit(___);}
#if 0
#line 103
{ } 
#endif
#line 122 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\surface_functions.h"
template< class T> 
#line 123
__declspec(deprecated) static __forceinline void surf1Dread(T *res, ::surface< void, 1>  surf, int x, int s, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)s;(void)mode;::exit(___);}
#if 0
#line 124
{ 
#line 128
} 
#endif
#line 130 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\surface_functions.h"
template< class T> 
#line 131
__declspec(deprecated) static __forceinline T surf1Dread(::surface< void, 1>  surf, int x, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)surf;(void)x;(void)mode;::exit(___);}
#if 0
#line 132
{ 
#line 138
} 
#endif
#line 140 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\surface_functions.h"
template< class T> 
#line 141
__declspec(deprecated) static __forceinline void surf1Dread(T *res, ::surface< void, 1>  surf, int x, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)mode;::exit(___);}
#if 0
#line 142
{ 
#line 146
} 
#endif
#line 149 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\surface_functions.h"
template< class T> 
#line 150
__declspec(deprecated) static __forceinline void surf2Dread(T *res, ::surface< void, 2>  surf, int x, int y, int s, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)y;(void)s;(void)mode;::exit(___);}
#if 0
#line 151
{ 
#line 155
} 
#endif
#line 157 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\surface_functions.h"
template< class T> 
#line 158
__declspec(deprecated) static __forceinline T surf2Dread(::surface< void, 2>  surf, int x, int y, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)surf;(void)x;(void)y;(void)mode;::exit(___);}
#if 0
#line 159
{ 
#line 165
} 
#endif
#line 167 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\surface_functions.h"
template< class T> 
#line 168
__declspec(deprecated) static __forceinline void surf2Dread(T *res, ::surface< void, 2>  surf, int x, int y, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)y;(void)mode;::exit(___);}
#if 0
#line 169
{ 
#line 173
} 
#endif
#line 176 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\surface_functions.h"
template< class T> 
#line 177
__declspec(deprecated) static __forceinline void surf3Dread(T *res, ::surface< void, 3>  surf, int x, int y, int z, int s, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)y;(void)z;(void)s;(void)mode;::exit(___);}
#if 0
#line 178
{ 
#line 182
} 
#endif
#line 184 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\surface_functions.h"
template< class T> 
#line 185
__declspec(deprecated) static __forceinline T surf3Dread(::surface< void, 3>  surf, int x, int y, int z, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)surf;(void)x;(void)y;(void)z;(void)mode;::exit(___);}
#if 0
#line 186
{ 
#line 192
} 
#endif
#line 194 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\surface_functions.h"
template< class T> 
#line 195
__declspec(deprecated) static __forceinline void surf3Dread(T *res, ::surface< void, 3>  surf, int x, int y, int z, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)y;(void)z;(void)mode;::exit(___);}
#if 0
#line 196
{ 
#line 200
} 
#endif
#line 204 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\surface_functions.h"
template< class T> 
#line 205
__declspec(deprecated) static __forceinline void surf1DLayeredread(T *res, ::surface< void, 241>  surf, int x, int layer, int s, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)layer;(void)s;(void)mode;::exit(___);}
#if 0
#line 206
{ 
#line 210
} 
#endif
#line 212 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\surface_functions.h"
template< class T> 
#line 213
__declspec(deprecated) static __forceinline T surf1DLayeredread(::surface< void, 241>  surf, int x, int layer, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)surf;(void)x;(void)layer;(void)mode;::exit(___);}
#if 0
#line 214
{ 
#line 220
} 
#endif
#line 223 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\surface_functions.h"
template< class T> 
#line 224
__declspec(deprecated) static __forceinline void surf1DLayeredread(T *res, ::surface< void, 241>  surf, int x, int layer, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)layer;(void)mode;::exit(___);}
#if 0
#line 225
{ 
#line 229
} 
#endif
#line 232 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\surface_functions.h"
template< class T> 
#line 233
__declspec(deprecated) static __forceinline void surf2DLayeredread(T *res, ::surface< void, 242>  surf, int x, int y, int layer, int s, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)y;(void)layer;(void)s;(void)mode;::exit(___);}
#if 0
#line 234
{ 
#line 238
} 
#endif
#line 240 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\surface_functions.h"
template< class T> 
#line 241
__declspec(deprecated) static __forceinline T surf2DLayeredread(::surface< void, 242>  surf, int x, int y, int layer, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)surf;(void)x;(void)y;(void)layer;(void)mode;::exit(___);}
#if 0
#line 242
{ 
#line 248
} 
#endif
#line 251 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\surface_functions.h"
template< class T> 
#line 252
__declspec(deprecated) static __forceinline void surf2DLayeredread(T *res, ::surface< void, 242>  surf, int x, int y, int layer, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)y;(void)layer;(void)mode;::exit(___);}
#if 0
#line 253
{ 
#line 257
} 
#endif
#line 260 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\surface_functions.h"
template< class T> static __forceinline void 
#line 261
surfCubemapread(T *res, ::surface< void, 12>  surf, int x, int y, int face, int s, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)y;(void)face;(void)s;(void)mode;::exit(___);}
#if 0
#line 262
{ 
#line 266
} 
#endif
#line 268 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\surface_functions.h"
template< class T> 
#line 269
__declspec(deprecated) static __forceinline T surfCubemapread(::surface< void, 12>  surf, int x, int y, int face, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)surf;(void)x;(void)y;(void)face;(void)mode;::exit(___);}
#if 0
#line 270
{ 
#line 277
} 
#endif
#line 279 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\surface_functions.h"
template< class T> 
#line 280
__declspec(deprecated) static __forceinline void surfCubemapread(T *res, ::surface< void, 12>  surf, int x, int y, int face, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)y;(void)face;(void)mode;::exit(___);}
#if 0
#line 281
{ 
#line 285
} 
#endif
#line 288 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\surface_functions.h"
template< class T> 
#line 289
__declspec(deprecated) static __forceinline void surfCubemapLayeredread(T *res, ::surface< void, 252>  surf, int x, int y, int layerFace, int s, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)y;(void)layerFace;(void)s;(void)mode;::exit(___);}
#if 0
#line 290
{ 
#line 294
} 
#endif
#line 296 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\surface_functions.h"
template< class T> 
#line 297
__declspec(deprecated) static __forceinline T surfCubemapLayeredread(::surface< void, 252>  surf, int x, int y, int layerFace, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)surf;(void)x;(void)y;(void)layerFace;(void)mode;::exit(___);}
#if 0
#line 298
{ 
#line 304
} 
#endif
#line 306 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\surface_functions.h"
template< class T> 
#line 307
__declspec(deprecated) static __forceinline void surfCubemapLayeredread(T *res, ::surface< void, 252>  surf, int x, int y, int layerFace, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)y;(void)layerFace;(void)mode;::exit(___);}
#if 0
#line 308
{ 
#line 312
} 
#endif
#line 315 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\surface_functions.h"
template< class T> 
#line 316
__declspec(deprecated) static __forceinline void surf1Dwrite(T val, ::surface< void, 1>  surf, int x, int s, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)s;(void)mode;::exit(___);}
#if 0
#line 317
{ 
#line 321
} 
#endif
#line 323 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\surface_functions.h"
template< class T> 
#line 324
__declspec(deprecated) static __forceinline void surf1Dwrite(T val, ::surface< void, 1>  surf, int x, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)mode;::exit(___);}
#if 0
#line 325
{ 
#line 329
} 
#endif
#line 333 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\surface_functions.h"
template< class T> 
#line 334
__declspec(deprecated) static __forceinline void surf2Dwrite(T val, ::surface< void, 2>  surf, int x, int y, int s, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)y;(void)s;(void)mode;::exit(___);}
#if 0
#line 335
{ 
#line 339
} 
#endif
#line 341 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\surface_functions.h"
template< class T> 
#line 342
__declspec(deprecated) static __forceinline void surf2Dwrite(T val, ::surface< void, 2>  surf, int x, int y, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)y;(void)mode;::exit(___);}
#if 0
#line 343
{ 
#line 347
} 
#endif
#line 350 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\surface_functions.h"
template< class T> 
#line 351
__declspec(deprecated) static __forceinline void surf3Dwrite(T val, ::surface< void, 3>  surf, int x, int y, int z, int s, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)y;(void)z;(void)s;(void)mode;::exit(___);}
#if 0
#line 352
{ 
#line 356
} 
#endif
#line 358 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\surface_functions.h"
template< class T> 
#line 359
__declspec(deprecated) static __forceinline void surf3Dwrite(T val, ::surface< void, 3>  surf, int x, int y, int z, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)y;(void)z;(void)mode;::exit(___);}
#if 0
#line 360
{ 
#line 364
} 
#endif
#line 367 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\surface_functions.h"
template< class T> 
#line 368
__declspec(deprecated) static __forceinline void surf1DLayeredwrite(T val, ::surface< void, 241>  surf, int x, int layer, int s, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)layer;(void)s;(void)mode;::exit(___);}
#if 0
#line 369
{ 
#line 373
} 
#endif
#line 375 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\surface_functions.h"
template< class T> 
#line 376
__declspec(deprecated) static __forceinline void surf1DLayeredwrite(T val, ::surface< void, 241>  surf, int x, int layer, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)layer;(void)mode;::exit(___);}
#if 0
#line 377
{ 
#line 381
} 
#endif
#line 384 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\surface_functions.h"
template< class T> 
#line 385
__declspec(deprecated) static __forceinline void surf2DLayeredwrite(T val, ::surface< void, 242>  surf, int x, int y, int layer, int s, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)y;(void)layer;(void)s;(void)mode;::exit(___);}
#if 0
#line 386
{ 
#line 390
} 
#endif
#line 392 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\surface_functions.h"
template< class T> 
#line 393
__declspec(deprecated) static __forceinline void surf2DLayeredwrite(T val, ::surface< void, 242>  surf, int x, int y, int layer, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)y;(void)layer;(void)mode;::exit(___);}
#if 0
#line 394
{ 
#line 398
} 
#endif
#line 401 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\surface_functions.h"
template< class T> 
#line 402
__declspec(deprecated) static __forceinline void surfCubemapwrite(T val, ::surface< void, 12>  surf, int x, int y, int face, int s, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)y;(void)face;(void)s;(void)mode;::exit(___);}
#if 0
#line 403
{ 
#line 407
} 
#endif
#line 409 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\surface_functions.h"
template< class T> 
#line 410
__declspec(deprecated) static __forceinline void surfCubemapwrite(T val, ::surface< void, 12>  surf, int x, int y, int face, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)y;(void)face;(void)mode;::exit(___);}
#if 0
#line 411
{ 
#line 415
} 
#endif
#line 419 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\surface_functions.h"
template< class T> 
#line 420
__declspec(deprecated) static __forceinline void surfCubemapLayeredwrite(T val, ::surface< void, 252>  surf, int x, int y, int layerFace, int s, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)y;(void)layerFace;(void)s;(void)mode;::exit(___);}
#if 0
#line 421
{ 
#line 425
} 
#endif
#line 427 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\surface_functions.h"
template< class T> 
#line 428
__declspec(deprecated) static __forceinline void surfCubemapLayeredwrite(T val, ::surface< void, 252>  surf, int x, int y, int layerFace, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)y;(void)layerFace;(void)mode;::exit(___);}
#if 0
#line 429
{ 
#line 433
} 
#endif
#line 72 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\texture_fetch_functions.h"
template< class T> 
#line 73
struct __nv_tex_rmet_ret { }; 
#line 75
template<> struct __nv_tex_rmet_ret< char>  { typedef char type; }; 
#line 76
template<> struct __nv_tex_rmet_ret< signed char>  { typedef signed char type; }; 
#line 77
template<> struct __nv_tex_rmet_ret< unsigned char>  { typedef unsigned char type; }; 
#line 78
template<> struct __nv_tex_rmet_ret< char1>  { typedef char1 type; }; 
#line 79
template<> struct __nv_tex_rmet_ret< uchar1>  { typedef uchar1 type; }; 
#line 80
template<> struct __nv_tex_rmet_ret< char2>  { typedef char2 type; }; 
#line 81
template<> struct __nv_tex_rmet_ret< uchar2>  { typedef uchar2 type; }; 
#line 82
template<> struct __nv_tex_rmet_ret< char4>  { typedef char4 type; }; 
#line 83
template<> struct __nv_tex_rmet_ret< uchar4>  { typedef uchar4 type; }; 
#line 85
template<> struct __nv_tex_rmet_ret< short>  { typedef short type; }; 
#line 86
template<> struct __nv_tex_rmet_ret< unsigned short>  { typedef unsigned short type; }; 
#line 87
template<> struct __nv_tex_rmet_ret< short1>  { typedef short1 type; }; 
#line 88
template<> struct __nv_tex_rmet_ret< ushort1>  { typedef ushort1 type; }; 
#line 89
template<> struct __nv_tex_rmet_ret< short2>  { typedef short2 type; }; 
#line 90
template<> struct __nv_tex_rmet_ret< ushort2>  { typedef ushort2 type; }; 
#line 91
template<> struct __nv_tex_rmet_ret< short4>  { typedef short4 type; }; 
#line 92
template<> struct __nv_tex_rmet_ret< ushort4>  { typedef ushort4 type; }; 
#line 94
template<> struct __nv_tex_rmet_ret< int>  { typedef int type; }; 
#line 95
template<> struct __nv_tex_rmet_ret< unsigned>  { typedef unsigned type; }; 
#line 96
template<> struct __nv_tex_rmet_ret< int1>  { typedef int1 type; }; 
#line 97
template<> struct __nv_tex_rmet_ret< uint1>  { typedef uint1 type; }; 
#line 98
template<> struct __nv_tex_rmet_ret< int2>  { typedef int2 type; }; 
#line 99
template<> struct __nv_tex_rmet_ret< uint2>  { typedef uint2 type; }; 
#line 100
template<> struct __nv_tex_rmet_ret< int4>  { typedef int4 type; }; 
#line 101
template<> struct __nv_tex_rmet_ret< uint4>  { typedef uint4 type; }; 
#line 104
template<> struct __nv_tex_rmet_ret< long>  { typedef long type; }; 
#line 105
template<> struct __nv_tex_rmet_ret< unsigned long>  { typedef unsigned long type; }; 
#line 106
template<> struct __nv_tex_rmet_ret< long1>  { typedef long1 type; }; 
#line 107
template<> struct __nv_tex_rmet_ret< ulong1>  { typedef ulong1 type; }; 
#line 108
template<> struct __nv_tex_rmet_ret< long2>  { typedef long2 type; }; 
#line 109
template<> struct __nv_tex_rmet_ret< ulong2>  { typedef ulong2 type; }; 
#line 110
template<> struct __nv_tex_rmet_ret< long4>  { typedef long4 type; }; 
#line 111
template<> struct __nv_tex_rmet_ret< ulong4>  { typedef ulong4 type; }; 
#line 113 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\texture_fetch_functions.h"
template<> struct __nv_tex_rmet_ret< float>  { typedef float type; }; 
#line 114
template<> struct __nv_tex_rmet_ret< float1>  { typedef float1 type; }; 
#line 115
template<> struct __nv_tex_rmet_ret< float2>  { typedef float2 type; }; 
#line 116
template<> struct __nv_tex_rmet_ret< float4>  { typedef float4 type; }; 
#line 119
template< class T> struct __nv_tex_rmet_cast { typedef T *type; }; 
#line 121
template<> struct __nv_tex_rmet_cast< long>  { typedef int *type; }; 
#line 122
template<> struct __nv_tex_rmet_cast< unsigned long>  { typedef unsigned *type; }; 
#line 123
template<> struct __nv_tex_rmet_cast< long1>  { typedef int1 *type; }; 
#line 124
template<> struct __nv_tex_rmet_cast< ulong1>  { typedef uint1 *type; }; 
#line 125
template<> struct __nv_tex_rmet_cast< long2>  { typedef int2 *type; }; 
#line 126
template<> struct __nv_tex_rmet_cast< ulong2>  { typedef uint2 *type; }; 
#line 127
template<> struct __nv_tex_rmet_cast< long4>  { typedef int4 *type; }; 
#line 128
template<> struct __nv_tex_rmet_cast< ulong4>  { typedef uint4 *type; }; 
#line 131 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\texture_fetch_functions.h"
template< class T> 
#line 132
__declspec(deprecated) static __forceinline typename __nv_tex_rmet_ret< T> ::type tex1Dfetch(texture< T, 1, cudaReadModeElementType>  t, int x) {int volatile ___ = 1;(void)t;(void)x;::exit(___);}
#if 0
#line 133
{ 
#line 139
} 
#endif
#line 141 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\texture_fetch_functions.h"
template< class T> 
#line 142
struct __nv_tex_rmnf_ret { }; 
#line 144
template<> struct __nv_tex_rmnf_ret< char>  { typedef float type; }; 
#line 145
template<> struct __nv_tex_rmnf_ret< signed char>  { typedef float type; }; 
#line 146
template<> struct __nv_tex_rmnf_ret< unsigned char>  { typedef float type; }; 
#line 147
template<> struct __nv_tex_rmnf_ret< short>  { typedef float type; }; 
#line 148
template<> struct __nv_tex_rmnf_ret< unsigned short>  { typedef float type; }; 
#line 149
template<> struct __nv_tex_rmnf_ret< char1>  { typedef float1 type; }; 
#line 150
template<> struct __nv_tex_rmnf_ret< uchar1>  { typedef float1 type; }; 
#line 151
template<> struct __nv_tex_rmnf_ret< short1>  { typedef float1 type; }; 
#line 152
template<> struct __nv_tex_rmnf_ret< ushort1>  { typedef float1 type; }; 
#line 153
template<> struct __nv_tex_rmnf_ret< char2>  { typedef float2 type; }; 
#line 154
template<> struct __nv_tex_rmnf_ret< uchar2>  { typedef float2 type; }; 
#line 155
template<> struct __nv_tex_rmnf_ret< short2>  { typedef float2 type; }; 
#line 156
template<> struct __nv_tex_rmnf_ret< ushort2>  { typedef float2 type; }; 
#line 157
template<> struct __nv_tex_rmnf_ret< char4>  { typedef float4 type; }; 
#line 158
template<> struct __nv_tex_rmnf_ret< uchar4>  { typedef float4 type; }; 
#line 159
template<> struct __nv_tex_rmnf_ret< short4>  { typedef float4 type; }; 
#line 160
template<> struct __nv_tex_rmnf_ret< ushort4>  { typedef float4 type; }; 
#line 162
template< class T> 
#line 163
__declspec(deprecated) static __forceinline typename __nv_tex_rmnf_ret< T> ::type tex1Dfetch(texture< T, 1, cudaReadModeNormalizedFloat>  t, int x) {int volatile ___ = 1;(void)t;(void)x;::exit(___);}
#if 0
#line 164
{ 
#line 171
} 
#endif
#line 174 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\texture_fetch_functions.h"
template< class T> 
#line 175
__declspec(deprecated) static __forceinline typename __nv_tex_rmet_ret< T> ::type tex1D(texture< T, 1, cudaReadModeElementType>  t, float x) {int volatile ___ = 1;(void)t;(void)x;::exit(___);}
#if 0
#line 176
{ 
#line 182
} 
#endif
#line 184 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\texture_fetch_functions.h"
template< class T> 
#line 185
__declspec(deprecated) static __forceinline typename __nv_tex_rmnf_ret< T> ::type tex1D(texture< T, 1, cudaReadModeNormalizedFloat>  t, float x) {int volatile ___ = 1;(void)t;(void)x;::exit(___);}
#if 0
#line 186
{ 
#line 193
} 
#endif
#line 197 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\texture_fetch_functions.h"
template< class T> 
#line 198
__declspec(deprecated) static __forceinline typename __nv_tex_rmet_ret< T> ::type tex2D(texture< T, 2, cudaReadModeElementType>  t, float x, float y) {int volatile ___ = 1;(void)t;(void)x;(void)y;::exit(___);}
#if 0
#line 199
{ 
#line 206
} 
#endif
#line 208 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\texture_fetch_functions.h"
template< class T> 
#line 209
__declspec(deprecated) static __forceinline typename __nv_tex_rmnf_ret< T> ::type tex2D(texture< T, 2, cudaReadModeNormalizedFloat>  t, float x, float y) {int volatile ___ = 1;(void)t;(void)x;(void)y;::exit(___);}
#if 0
#line 210
{ 
#line 217
} 
#endif
#line 221 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\texture_fetch_functions.h"
template< class T> 
#line 222
__declspec(deprecated) static __forceinline typename __nv_tex_rmet_ret< T> ::type tex1DLayered(texture< T, 241, cudaReadModeElementType>  t, float x, int layer) {int volatile ___ = 1;(void)t;(void)x;(void)layer;::exit(___);}
#if 0
#line 223
{ 
#line 229
} 
#endif
#line 231 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\texture_fetch_functions.h"
template< class T> 
#line 232
__declspec(deprecated) static __forceinline typename __nv_tex_rmnf_ret< T> ::type tex1DLayered(texture< T, 241, cudaReadModeNormalizedFloat>  t, float x, int layer) {int volatile ___ = 1;(void)t;(void)x;(void)layer;::exit(___);}
#if 0
#line 233
{ 
#line 240
} 
#endif
#line 244 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\texture_fetch_functions.h"
template< class T> 
#line 245
__declspec(deprecated) static __forceinline typename __nv_tex_rmet_ret< T> ::type tex2DLayered(texture< T, 242, cudaReadModeElementType>  t, float x, float y, int layer) {int volatile ___ = 1;(void)t;(void)x;(void)y;(void)layer;::exit(___);}
#if 0
#line 246
{ 
#line 252
} 
#endif
#line 254 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\texture_fetch_functions.h"
template< class T> 
#line 255
__declspec(deprecated) static __forceinline typename __nv_tex_rmnf_ret< T> ::type tex2DLayered(texture< T, 242, cudaReadModeNormalizedFloat>  t, float x, float y, int layer) {int volatile ___ = 1;(void)t;(void)x;(void)y;(void)layer;::exit(___);}
#if 0
#line 256
{ 
#line 263
} 
#endif
#line 266 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\texture_fetch_functions.h"
template< class T> 
#line 267
__declspec(deprecated) static __forceinline typename __nv_tex_rmet_ret< T> ::type tex3D(texture< T, 3, cudaReadModeElementType>  t, float x, float y, float z) {int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;::exit(___);}
#if 0
#line 268
{ 
#line 274
} 
#endif
#line 276 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\texture_fetch_functions.h"
template< class T> 
#line 277
__declspec(deprecated) static __forceinline typename __nv_tex_rmnf_ret< T> ::type tex3D(texture< T, 3, cudaReadModeNormalizedFloat>  t, float x, float y, float z) {int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;::exit(___);}
#if 0
#line 278
{ 
#line 285
} 
#endif
#line 288 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\texture_fetch_functions.h"
template< class T> 
#line 289
__declspec(deprecated) static __forceinline typename __nv_tex_rmet_ret< T> ::type texCubemap(texture< T, 12, cudaReadModeElementType>  t, float x, float y, float z) {int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;::exit(___);}
#if 0
#line 290
{ 
#line 296
} 
#endif
#line 298 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\texture_fetch_functions.h"
template< class T> 
#line 299
__declspec(deprecated) static __forceinline typename __nv_tex_rmnf_ret< T> ::type texCubemap(texture< T, 12, cudaReadModeNormalizedFloat>  t, float x, float y, float z) {int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;::exit(___);}
#if 0
#line 300
{ 
#line 307
} 
#endif
#line 310 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\texture_fetch_functions.h"
template< class T> 
#line 311
struct __nv_tex2dgather_ret { }; 
#line 312
template<> struct __nv_tex2dgather_ret< char>  { typedef char4 type; }; 
#line 313
template<> struct __nv_tex2dgather_ret< signed char>  { typedef char4 type; }; 
#line 314
template<> struct __nv_tex2dgather_ret< char1>  { typedef char4 type; }; 
#line 315
template<> struct __nv_tex2dgather_ret< char2>  { typedef char4 type; }; 
#line 316
template<> struct __nv_tex2dgather_ret< char3>  { typedef char4 type; }; 
#line 317
template<> struct __nv_tex2dgather_ret< char4>  { typedef char4 type; }; 
#line 318
template<> struct __nv_tex2dgather_ret< unsigned char>  { typedef uchar4 type; }; 
#line 319
template<> struct __nv_tex2dgather_ret< uchar1>  { typedef uchar4 type; }; 
#line 320
template<> struct __nv_tex2dgather_ret< uchar2>  { typedef uchar4 type; }; 
#line 321
template<> struct __nv_tex2dgather_ret< uchar3>  { typedef uchar4 type; }; 
#line 322
template<> struct __nv_tex2dgather_ret< uchar4>  { typedef uchar4 type; }; 
#line 324
template<> struct __nv_tex2dgather_ret< short>  { typedef short4 type; }; 
#line 325
template<> struct __nv_tex2dgather_ret< short1>  { typedef short4 type; }; 
#line 326
template<> struct __nv_tex2dgather_ret< short2>  { typedef short4 type; }; 
#line 327
template<> struct __nv_tex2dgather_ret< short3>  { typedef short4 type; }; 
#line 328
template<> struct __nv_tex2dgather_ret< short4>  { typedef short4 type; }; 
#line 329
template<> struct __nv_tex2dgather_ret< unsigned short>  { typedef ushort4 type; }; 
#line 330
template<> struct __nv_tex2dgather_ret< ushort1>  { typedef ushort4 type; }; 
#line 331
template<> struct __nv_tex2dgather_ret< ushort2>  { typedef ushort4 type; }; 
#line 332
template<> struct __nv_tex2dgather_ret< ushort3>  { typedef ushort4 type; }; 
#line 333
template<> struct __nv_tex2dgather_ret< ushort4>  { typedef ushort4 type; }; 
#line 335
template<> struct __nv_tex2dgather_ret< int>  { typedef int4 type; }; 
#line 336
template<> struct __nv_tex2dgather_ret< int1>  { typedef int4 type; }; 
#line 337
template<> struct __nv_tex2dgather_ret< int2>  { typedef int4 type; }; 
#line 338
template<> struct __nv_tex2dgather_ret< int3>  { typedef int4 type; }; 
#line 339
template<> struct __nv_tex2dgather_ret< int4>  { typedef int4 type; }; 
#line 340
template<> struct __nv_tex2dgather_ret< unsigned>  { typedef uint4 type; }; 
#line 341
template<> struct __nv_tex2dgather_ret< uint1>  { typedef uint4 type; }; 
#line 342
template<> struct __nv_tex2dgather_ret< uint2>  { typedef uint4 type; }; 
#line 343
template<> struct __nv_tex2dgather_ret< uint3>  { typedef uint4 type; }; 
#line 344
template<> struct __nv_tex2dgather_ret< uint4>  { typedef uint4 type; }; 
#line 346
template<> struct __nv_tex2dgather_ret< float>  { typedef float4 type; }; 
#line 347
template<> struct __nv_tex2dgather_ret< float1>  { typedef float4 type; }; 
#line 348
template<> struct __nv_tex2dgather_ret< float2>  { typedef float4 type; }; 
#line 349
template<> struct __nv_tex2dgather_ret< float3>  { typedef float4 type; }; 
#line 350
template<> struct __nv_tex2dgather_ret< float4>  { typedef float4 type; }; 
#line 352
template< class T> static __forceinline typename __nv_tex2dgather_ret< T> ::type 
#line 353
tex2Dgather(texture< T, 2, cudaReadModeElementType>  t, float x, float y, int comp = 0) {int volatile ___ = 1;(void)t;(void)x;(void)y;(void)comp;::exit(___);}
#if 0
#line 354
{ 
#line 361
} 
#endif
#line 364 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\texture_fetch_functions.h"
template< class T> struct __nv_tex2dgather_rmnf_ret { }; 
#line 365
template<> struct __nv_tex2dgather_rmnf_ret< char>  { typedef float4 type; }; 
#line 366
template<> struct __nv_tex2dgather_rmnf_ret< signed char>  { typedef float4 type; }; 
#line 367
template<> struct __nv_tex2dgather_rmnf_ret< unsigned char>  { typedef float4 type; }; 
#line 368
template<> struct __nv_tex2dgather_rmnf_ret< char1>  { typedef float4 type; }; 
#line 369
template<> struct __nv_tex2dgather_rmnf_ret< uchar1>  { typedef float4 type; }; 
#line 370
template<> struct __nv_tex2dgather_rmnf_ret< char2>  { typedef float4 type; }; 
#line 371
template<> struct __nv_tex2dgather_rmnf_ret< uchar2>  { typedef float4 type; }; 
#line 372
template<> struct __nv_tex2dgather_rmnf_ret< char3>  { typedef float4 type; }; 
#line 373
template<> struct __nv_tex2dgather_rmnf_ret< uchar3>  { typedef float4 type; }; 
#line 374
template<> struct __nv_tex2dgather_rmnf_ret< char4>  { typedef float4 type; }; 
#line 375
template<> struct __nv_tex2dgather_rmnf_ret< uchar4>  { typedef float4 type; }; 
#line 376
template<> struct __nv_tex2dgather_rmnf_ret< signed short>  { typedef float4 type; }; 
#line 377
template<> struct __nv_tex2dgather_rmnf_ret< unsigned short>  { typedef float4 type; }; 
#line 378
template<> struct __nv_tex2dgather_rmnf_ret< short1>  { typedef float4 type; }; 
#line 379
template<> struct __nv_tex2dgather_rmnf_ret< ushort1>  { typedef float4 type; }; 
#line 380
template<> struct __nv_tex2dgather_rmnf_ret< short2>  { typedef float4 type; }; 
#line 381
template<> struct __nv_tex2dgather_rmnf_ret< ushort2>  { typedef float4 type; }; 
#line 382
template<> struct __nv_tex2dgather_rmnf_ret< short3>  { typedef float4 type; }; 
#line 383
template<> struct __nv_tex2dgather_rmnf_ret< ushort3>  { typedef float4 type; }; 
#line 384
template<> struct __nv_tex2dgather_rmnf_ret< short4>  { typedef float4 type; }; 
#line 385
template<> struct __nv_tex2dgather_rmnf_ret< ushort4>  { typedef float4 type; }; 
#line 387
template< class T> static __forceinline typename __nv_tex2dgather_rmnf_ret< T> ::type 
#line 388
tex2Dgather(texture< T, 2, cudaReadModeNormalizedFloat>  t, float x, float y, int comp = 0) {int volatile ___ = 1;(void)t;(void)x;(void)y;(void)comp;::exit(___);}
#if 0
#line 389
{ 
#line 396
} 
#endif
#line 400 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\texture_fetch_functions.h"
template< class T> 
#line 401
__declspec(deprecated) static __forceinline typename __nv_tex_rmet_ret< T> ::type tex1DLod(texture< T, 1, cudaReadModeElementType>  t, float x, float level) {int volatile ___ = 1;(void)t;(void)x;(void)level;::exit(___);}
#if 0
#line 402
{ 
#line 408
} 
#endif
#line 410 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\texture_fetch_functions.h"
template< class T> 
#line 411
__declspec(deprecated) static __forceinline typename __nv_tex_rmnf_ret< T> ::type tex1DLod(texture< T, 1, cudaReadModeNormalizedFloat>  t, float x, float level) {int volatile ___ = 1;(void)t;(void)x;(void)level;::exit(___);}
#if 0
#line 412
{ 
#line 419
} 
#endif
#line 422 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\texture_fetch_functions.h"
template< class T> 
#line 423
__declspec(deprecated) static __forceinline typename __nv_tex_rmet_ret< T> ::type tex2DLod(texture< T, 2, cudaReadModeElementType>  t, float x, float y, float level) {int volatile ___ = 1;(void)t;(void)x;(void)y;(void)level;::exit(___);}
#if 0
#line 424
{ 
#line 430
} 
#endif
#line 432 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\texture_fetch_functions.h"
template< class T> 
#line 433
__declspec(deprecated) static __forceinline typename __nv_tex_rmnf_ret< T> ::type tex2DLod(texture< T, 2, cudaReadModeNormalizedFloat>  t, float x, float y, float level) {int volatile ___ = 1;(void)t;(void)x;(void)y;(void)level;::exit(___);}
#if 0
#line 434
{ 
#line 441
} 
#endif
#line 444 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\texture_fetch_functions.h"
template< class T> 
#line 445
__declspec(deprecated) static __forceinline typename __nv_tex_rmet_ret< T> ::type tex1DLayeredLod(texture< T, 241, cudaReadModeElementType>  t, float x, int layer, float level) {int volatile ___ = 1;(void)t;(void)x;(void)layer;(void)level;::exit(___);}
#if 0
#line 446
{ 
#line 452
} 
#endif
#line 454 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\texture_fetch_functions.h"
template< class T> 
#line 455
__declspec(deprecated) static __forceinline typename __nv_tex_rmnf_ret< T> ::type tex1DLayeredLod(texture< T, 241, cudaReadModeNormalizedFloat>  t, float x, int layer, float level) {int volatile ___ = 1;(void)t;(void)x;(void)layer;(void)level;::exit(___);}
#if 0
#line 456
{ 
#line 463
} 
#endif
#line 466 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\texture_fetch_functions.h"
template< class T> 
#line 467
__declspec(deprecated) static __forceinline typename __nv_tex_rmet_ret< T> ::type tex2DLayeredLod(texture< T, 242, cudaReadModeElementType>  t, float x, float y, int layer, float level) {int volatile ___ = 1;(void)t;(void)x;(void)y;(void)layer;(void)level;::exit(___);}
#if 0
#line 468
{ 
#line 474
} 
#endif
#line 476 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\texture_fetch_functions.h"
template< class T> 
#line 477
__declspec(deprecated) static __forceinline typename __nv_tex_rmnf_ret< T> ::type tex2DLayeredLod(texture< T, 242, cudaReadModeNormalizedFloat>  t, float x, float y, int layer, float level) {int volatile ___ = 1;(void)t;(void)x;(void)y;(void)layer;(void)level;::exit(___);}
#if 0
#line 478
{ 
#line 485
} 
#endif
#line 488 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\texture_fetch_functions.h"
template< class T> 
#line 489
__declspec(deprecated) static __forceinline typename __nv_tex_rmet_ret< T> ::type tex3DLod(texture< T, 3, cudaReadModeElementType>  t, float x, float y, float z, float level) {int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;(void)level;::exit(___);}
#if 0
#line 490
{ 
#line 496
} 
#endif
#line 498 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\texture_fetch_functions.h"
template< class T> 
#line 499
__declspec(deprecated) static __forceinline typename __nv_tex_rmnf_ret< T> ::type tex3DLod(texture< T, 3, cudaReadModeNormalizedFloat>  t, float x, float y, float z, float level) {int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;(void)level;::exit(___);}
#if 0
#line 500
{ 
#line 507
} 
#endif
#line 510 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\texture_fetch_functions.h"
template< class T> 
#line 511
__declspec(deprecated) static __forceinline typename __nv_tex_rmet_ret< T> ::type texCubemapLod(texture< T, 12, cudaReadModeElementType>  t, float x, float y, float z, float level) {int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;(void)level;::exit(___);}
#if 0
#line 512
{ 
#line 518
} 
#endif
#line 520 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\texture_fetch_functions.h"
template< class T> 
#line 521
__declspec(deprecated) static __forceinline typename __nv_tex_rmnf_ret< T> ::type texCubemapLod(texture< T, 12, cudaReadModeNormalizedFloat>  t, float x, float y, float z, float level) {int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;(void)level;::exit(___);}
#if 0
#line 522
{ 
#line 529
} 
#endif
#line 533 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\texture_fetch_functions.h"
template< class T> 
#line 534
__declspec(deprecated) static __forceinline typename __nv_tex_rmet_ret< T> ::type texCubemapLayered(texture< T, 252, cudaReadModeElementType>  t, float x, float y, float z, int layer) {int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;(void)layer;::exit(___);}
#if 0
#line 535
{ 
#line 541
} 
#endif
#line 543 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\texture_fetch_functions.h"
template< class T> 
#line 544
__declspec(deprecated) static __forceinline typename __nv_tex_rmnf_ret< T> ::type texCubemapLayered(texture< T, 252, cudaReadModeNormalizedFloat>  t, float x, float y, float z, int layer) {int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;(void)layer;::exit(___);}
#if 0
#line 545
{ 
#line 552
} 
#endif
#line 556 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\texture_fetch_functions.h"
template< class T> 
#line 557
__declspec(deprecated) static __forceinline typename __nv_tex_rmet_ret< T> ::type texCubemapLayeredLod(texture< T, 252, cudaReadModeElementType>  t, float x, float y, float z, int layer, float level) {int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;(void)layer;(void)level;::exit(___);}
#if 0
#line 558
{ 
#line 564
} 
#endif
#line 566 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\texture_fetch_functions.h"
template< class T> 
#line 567
__declspec(deprecated) static __forceinline typename __nv_tex_rmnf_ret< T> ::type texCubemapLayeredLod(texture< T, 252, cudaReadModeNormalizedFloat>  t, float x, float y, float z, int layer, float level) {int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;(void)layer;(void)level;::exit(___);}
#if 0
#line 568
{ 
#line 575
} 
#endif
#line 579 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\texture_fetch_functions.h"
template< class T> 
#line 580
__declspec(deprecated) static __forceinline typename __nv_tex_rmet_ret< T> ::type texCubemapGrad(texture< T, 12, cudaReadModeElementType>  t, float x, float y, float z, ::float4 dPdx, ::float4 dPdy) {int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;(void)dPdx;(void)dPdy;::exit(___);}
#if 0
#line 581
{ 
#line 587
} 
#endif
#line 589 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\texture_fetch_functions.h"
template< class T> 
#line 590
__declspec(deprecated) static __forceinline typename __nv_tex_rmnf_ret< T> ::type texCubemapGrad(texture< T, 12, cudaReadModeNormalizedFloat>  t, float x, float y, float z, ::float4 dPdx, ::float4 dPdy) {int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;(void)dPdx;(void)dPdy;::exit(___);}
#if 0
#line 591
{ 
#line 598
} 
#endif
#line 602 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\texture_fetch_functions.h"
template< class T> 
#line 603
__declspec(deprecated) static __forceinline typename __nv_tex_rmet_ret< T> ::type texCubemapLayeredGrad(texture< T, 252, cudaReadModeElementType>  t, float x, float y, float z, int layer, ::float4 dPdx, ::float4 dPdy) {int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;(void)layer;(void)dPdx;(void)dPdy;::exit(___);}
#if 0
#line 604
{ 
#line 610
} 
#endif
#line 612 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\texture_fetch_functions.h"
template< class T> 
#line 613
__declspec(deprecated) static __forceinline typename __nv_tex_rmnf_ret< T> ::type texCubemapLayeredGrad(texture< T, 252, cudaReadModeNormalizedFloat>  t, float x, float y, float z, int layer, ::float4 dPdx, ::float4 dPdy) {int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;(void)layer;(void)dPdx;(void)dPdy;::exit(___);}
#if 0
#line 614
{ 
#line 621
} 
#endif
#line 625 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\texture_fetch_functions.h"
template< class T> 
#line 626
__declspec(deprecated) static __forceinline typename __nv_tex_rmet_ret< T> ::type tex1DGrad(texture< T, 1, cudaReadModeElementType>  t, float x, float dPdx, float dPdy) {int volatile ___ = 1;(void)t;(void)x;(void)dPdx;(void)dPdy;::exit(___);}
#if 0
#line 627
{ 
#line 633
} 
#endif
#line 635 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\texture_fetch_functions.h"
template< class T> 
#line 636
__declspec(deprecated) static __forceinline typename __nv_tex_rmnf_ret< T> ::type tex1DGrad(texture< T, 1, cudaReadModeNormalizedFloat>  t, float x, float dPdx, float dPdy) {int volatile ___ = 1;(void)t;(void)x;(void)dPdx;(void)dPdy;::exit(___);}
#if 0
#line 637
{ 
#line 644
} 
#endif
#line 648 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\texture_fetch_functions.h"
template< class T> 
#line 649
__declspec(deprecated) static __forceinline typename __nv_tex_rmet_ret< T> ::type tex2DGrad(texture< T, 2, cudaReadModeElementType>  t, float x, float y, ::float2 dPdx, ::float2 dPdy) {int volatile ___ = 1;(void)t;(void)x;(void)y;(void)dPdx;(void)dPdy;::exit(___);}
#if 0
#line 650
{ 
#line 656
} 
#endif
#line 658 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\texture_fetch_functions.h"
template< class T> 
#line 659
__declspec(deprecated) static __forceinline typename __nv_tex_rmnf_ret< T> ::type tex2DGrad(texture< T, 2, cudaReadModeNormalizedFloat>  t, float x, float y, ::float2 dPdx, ::float2 dPdy) {int volatile ___ = 1;(void)t;(void)x;(void)y;(void)dPdx;(void)dPdy;::exit(___);}
#if 0
#line 660
{ 
#line 667
} 
#endif
#line 670 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\texture_fetch_functions.h"
template< class T> 
#line 671
__declspec(deprecated) static __forceinline typename __nv_tex_rmet_ret< T> ::type tex1DLayeredGrad(texture< T, 241, cudaReadModeElementType>  t, float x, int layer, float dPdx, float dPdy) {int volatile ___ = 1;(void)t;(void)x;(void)layer;(void)dPdx;(void)dPdy;::exit(___);}
#if 0
#line 672
{ 
#line 678
} 
#endif
#line 680 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\texture_fetch_functions.h"
template< class T> 
#line 681
__declspec(deprecated) static __forceinline typename __nv_tex_rmnf_ret< T> ::type tex1DLayeredGrad(texture< T, 241, cudaReadModeNormalizedFloat>  t, float x, int layer, float dPdx, float dPdy) {int volatile ___ = 1;(void)t;(void)x;(void)layer;(void)dPdx;(void)dPdy;::exit(___);}
#if 0
#line 682
{ 
#line 689
} 
#endif
#line 692 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\texture_fetch_functions.h"
template< class T> 
#line 693
__declspec(deprecated) static __forceinline typename __nv_tex_rmet_ret< T> ::type tex2DLayeredGrad(texture< T, 242, cudaReadModeElementType>  t, float x, float y, int layer, ::float2 dPdx, ::float2 dPdy) {int volatile ___ = 1;(void)t;(void)x;(void)y;(void)layer;(void)dPdx;(void)dPdy;::exit(___);}
#if 0
#line 694
{ 
#line 700
} 
#endif
#line 702 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\texture_fetch_functions.h"
template< class T> 
#line 703
__declspec(deprecated) static __forceinline typename __nv_tex_rmnf_ret< T> ::type tex2DLayeredGrad(texture< T, 242, cudaReadModeNormalizedFloat>  t, float x, float y, int layer, ::float2 dPdx, ::float2 dPdy) {int volatile ___ = 1;(void)t;(void)x;(void)y;(void)layer;(void)dPdx;(void)dPdy;::exit(___);}
#if 0
#line 704
{ 
#line 711
} 
#endif
#line 714 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\texture_fetch_functions.h"
template< class T> 
#line 715
__declspec(deprecated) static __forceinline typename __nv_tex_rmet_ret< T> ::type tex3DGrad(texture< T, 3, cudaReadModeElementType>  t, float x, float y, float z, ::float4 dPdx, ::float4 dPdy) {int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;(void)dPdx;(void)dPdy;::exit(___);}
#if 0
#line 716
{ 
#line 722
} 
#endif
#line 724 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\texture_fetch_functions.h"
template< class T> 
#line 725
__declspec(deprecated) static __forceinline typename __nv_tex_rmnf_ret< T> ::type tex3DGrad(texture< T, 3, cudaReadModeNormalizedFloat>  t, float x, float y, float z, ::float4 dPdx, ::float4 dPdy) {int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;(void)dPdx;(void)dPdy;::exit(___);}
#if 0
#line 726
{ 
#line 733
} 
#endif
#line 64 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\texture_indirect_functions.h"
template< class T> struct __nv_itex_trait { }; 
#line 65
template<> struct __nv_itex_trait< char>  { typedef void type; }; 
#line 66
template<> struct __nv_itex_trait< signed char>  { typedef void type; }; 
#line 67
template<> struct __nv_itex_trait< char1>  { typedef void type; }; 
#line 68
template<> struct __nv_itex_trait< char2>  { typedef void type; }; 
#line 69
template<> struct __nv_itex_trait< char4>  { typedef void type; }; 
#line 70
template<> struct __nv_itex_trait< unsigned char>  { typedef void type; }; 
#line 71
template<> struct __nv_itex_trait< uchar1>  { typedef void type; }; 
#line 72
template<> struct __nv_itex_trait< uchar2>  { typedef void type; }; 
#line 73
template<> struct __nv_itex_trait< uchar4>  { typedef void type; }; 
#line 74
template<> struct __nv_itex_trait< short>  { typedef void type; }; 
#line 75
template<> struct __nv_itex_trait< short1>  { typedef void type; }; 
#line 76
template<> struct __nv_itex_trait< short2>  { typedef void type; }; 
#line 77
template<> struct __nv_itex_trait< short4>  { typedef void type; }; 
#line 78
template<> struct __nv_itex_trait< unsigned short>  { typedef void type; }; 
#line 79
template<> struct __nv_itex_trait< ushort1>  { typedef void type; }; 
#line 80
template<> struct __nv_itex_trait< ushort2>  { typedef void type; }; 
#line 81
template<> struct __nv_itex_trait< ushort4>  { typedef void type; }; 
#line 82
template<> struct __nv_itex_trait< int>  { typedef void type; }; 
#line 83
template<> struct __nv_itex_trait< int1>  { typedef void type; }; 
#line 84
template<> struct __nv_itex_trait< int2>  { typedef void type; }; 
#line 85
template<> struct __nv_itex_trait< int4>  { typedef void type; }; 
#line 86
template<> struct __nv_itex_trait< unsigned>  { typedef void type; }; 
#line 87
template<> struct __nv_itex_trait< uint1>  { typedef void type; }; 
#line 88
template<> struct __nv_itex_trait< uint2>  { typedef void type; }; 
#line 89
template<> struct __nv_itex_trait< uint4>  { typedef void type; }; 
#line 91
template<> struct __nv_itex_trait< long>  { typedef void type; }; 
#line 92
template<> struct __nv_itex_trait< long1>  { typedef void type; }; 
#line 93
template<> struct __nv_itex_trait< long2>  { typedef void type; }; 
#line 94
template<> struct __nv_itex_trait< long4>  { typedef void type; }; 
#line 95
template<> struct __nv_itex_trait< unsigned long>  { typedef void type; }; 
#line 96
template<> struct __nv_itex_trait< ulong1>  { typedef void type; }; 
#line 97
template<> struct __nv_itex_trait< ulong2>  { typedef void type; }; 
#line 98
template<> struct __nv_itex_trait< ulong4>  { typedef void type; }; 
#line 100 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\texture_indirect_functions.h"
template<> struct __nv_itex_trait< float>  { typedef void type; }; 
#line 101
template<> struct __nv_itex_trait< float1>  { typedef void type; }; 
#line 102
template<> struct __nv_itex_trait< float2>  { typedef void type; }; 
#line 103
template<> struct __nv_itex_trait< float4>  { typedef void type; }; 
#line 107
template< class T> static typename __nv_itex_trait< T> ::type 
#line 108
tex1Dfetch(T *ptr, ::cudaTextureObject_t obj, int x) {int volatile ___ = 1;(void)ptr;(void)obj;(void)x;::exit(___);}
#if 0
#line 109
{ 
#line 113
} 
#endif
#line 115 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 116
tex1Dfetch(::cudaTextureObject_t texObject, int x) {int volatile ___ = 1;(void)texObject;(void)x;::exit(___);}
#if 0
#line 117
{ 
#line 123
} 
#endif
#line 125 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\texture_indirect_functions.h"
template< class T> static typename __nv_itex_trait< T> ::type 
#line 126
tex1D(T *ptr, ::cudaTextureObject_t obj, float x) {int volatile ___ = 1;(void)ptr;(void)obj;(void)x;::exit(___);}
#if 0
#line 127
{ 
#line 131
} 
#endif
#line 134 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 135
tex1D(::cudaTextureObject_t texObject, float x) {int volatile ___ = 1;(void)texObject;(void)x;::exit(___);}
#if 0
#line 136
{ 
#line 142
} 
#endif
#line 145 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\texture_indirect_functions.h"
template< class T> static typename __nv_itex_trait< T> ::type 
#line 146
tex2D(T *ptr, ::cudaTextureObject_t obj, float x, float y) {int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;::exit(___);}
#if 0
#line 147
{ 
#line 151
} 
#endif
#line 153 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 154
tex2D(::cudaTextureObject_t texObject, float x, float y) {int volatile ___ = 1;(void)texObject;(void)x;(void)y;::exit(___);}
#if 0
#line 155
{ 
#line 161
} 
#endif
#line 164 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\texture_indirect_functions.h"
template< class T> static typename __nv_itex_trait< T> ::type 
#line 165
tex2D(T *ptr, ::cudaTextureObject_t obj, float x, float y, bool *
#line 166
isResident) {int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)isResident;::exit(___);}
#if 0
#line 167
{ 
#line 173
} 
#endif
#line 175 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 176
tex2D(::cudaTextureObject_t texObject, float x, float y, bool *isResident) {int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)isResident;::exit(___);}
#if 0
#line 177
{ 
#line 183
} 
#endif
#line 188 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\texture_indirect_functions.h"
template< class T> static typename __nv_itex_trait< T> ::type 
#line 189
tex3D(T *ptr, ::cudaTextureObject_t obj, float x, float y, float z) {int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;::exit(___);}
#if 0
#line 190
{ 
#line 194
} 
#endif
#line 196 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 197
tex3D(::cudaTextureObject_t texObject, float x, float y, float z) {int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;::exit(___);}
#if 0
#line 198
{ 
#line 204
} 
#endif
#line 207 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\texture_indirect_functions.h"
template< class T> static typename __nv_itex_trait< T> ::type 
#line 208
tex3D(T *ptr, ::cudaTextureObject_t obj, float x, float y, float z, bool *
#line 209
isResident) {int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;(void)isResident;::exit(___);}
#if 0
#line 210
{ 
#line 216
} 
#endif
#line 218 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 219
tex3D(::cudaTextureObject_t texObject, float x, float y, float z, bool *isResident) {int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;(void)isResident;::exit(___);}
#if 0
#line 220
{ 
#line 226
} 
#endif
#line 230 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\texture_indirect_functions.h"
template< class T> static typename __nv_itex_trait< T> ::type 
#line 231
tex1DLayered(T *ptr, ::cudaTextureObject_t obj, float x, int layer) {int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)layer;::exit(___);}
#if 0
#line 232
{ 
#line 236
} 
#endif
#line 238 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 239
tex1DLayered(::cudaTextureObject_t texObject, float x, int layer) {int volatile ___ = 1;(void)texObject;(void)x;(void)layer;::exit(___);}
#if 0
#line 240
{ 
#line 246
} 
#endif
#line 248 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\texture_indirect_functions.h"
template< class T> static typename __nv_itex_trait< T> ::type 
#line 249
tex2DLayered(T *ptr, ::cudaTextureObject_t obj, float x, float y, int layer) {int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)layer;::exit(___);}
#if 0
#line 250
{ 
#line 254
} 
#endif
#line 256 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 257
tex2DLayered(::cudaTextureObject_t texObject, float x, float y, int layer) {int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)layer;::exit(___);}
#if 0
#line 258
{ 
#line 264
} 
#endif
#line 267 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\texture_indirect_functions.h"
template< class T> static typename __nv_itex_trait< T> ::type 
#line 268
tex2DLayered(T *ptr, ::cudaTextureObject_t obj, float x, float y, int layer, bool *isResident) {int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)layer;(void)isResident;::exit(___);}
#if 0
#line 269
{ 
#line 275
} 
#endif
#line 277 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 278
tex2DLayered(::cudaTextureObject_t texObject, float x, float y, int layer, bool *isResident) {int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)layer;(void)isResident;::exit(___);}
#if 0
#line 279
{ 
#line 285
} 
#endif
#line 289 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\texture_indirect_functions.h"
template< class T> static typename __nv_itex_trait< T> ::type 
#line 290
texCubemap(T *ptr, ::cudaTextureObject_t obj, float x, float y, float z) {int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;::exit(___);}
#if 0
#line 291
{ 
#line 295
} 
#endif
#line 298 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 299
texCubemap(::cudaTextureObject_t texObject, float x, float y, float z) {int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;::exit(___);}
#if 0
#line 300
{ 
#line 306
} 
#endif
#line 309 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\texture_indirect_functions.h"
template< class T> static typename __nv_itex_trait< T> ::type 
#line 310
texCubemapLayered(T *ptr, ::cudaTextureObject_t obj, float x, float y, float z, int layer) {int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;(void)layer;::exit(___);}
#if 0
#line 311
{ 
#line 315
} 
#endif
#line 317 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 318
texCubemapLayered(::cudaTextureObject_t texObject, float x, float y, float z, int layer) {int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;(void)layer;::exit(___);}
#if 0
#line 319
{ 
#line 325
} 
#endif
#line 327 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\texture_indirect_functions.h"
template< class T> static typename __nv_itex_trait< T> ::type 
#line 328
tex2Dgather(T *ptr, ::cudaTextureObject_t obj, float x, float y, int comp = 0) {int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)comp;::exit(___);}
#if 0
#line 329
{ 
#line 333
} 
#endif
#line 335 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 336
tex2Dgather(::cudaTextureObject_t to, float x, float y, int comp = 0) {int volatile ___ = 1;(void)to;(void)x;(void)y;(void)comp;::exit(___);}
#if 0
#line 337
{ 
#line 343
} 
#endif
#line 346 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\texture_indirect_functions.h"
template< class T> static typename __nv_itex_trait< T> ::type 
#line 347
tex2Dgather(T *ptr, ::cudaTextureObject_t obj, float x, float y, bool *isResident, int comp = 0) {int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)isResident;(void)comp;::exit(___);}
#if 0
#line 348
{ 
#line 354
} 
#endif
#line 356 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 357
tex2Dgather(::cudaTextureObject_t to, float x, float y, bool *isResident, int comp = 0) {int volatile ___ = 1;(void)to;(void)x;(void)y;(void)isResident;(void)comp;::exit(___);}
#if 0
#line 358
{ 
#line 364
} 
#endif
#line 368 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\texture_indirect_functions.h"
template< class T> static typename __nv_itex_trait< T> ::type 
#line 369
tex1DLod(T *ptr, ::cudaTextureObject_t obj, float x, float level) {int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)level;::exit(___);}
#if 0
#line 370
{ 
#line 374
} 
#endif
#line 376 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 377
tex1DLod(::cudaTextureObject_t texObject, float x, float level) {int volatile ___ = 1;(void)texObject;(void)x;(void)level;::exit(___);}
#if 0
#line 378
{ 
#line 384
} 
#endif
#line 387 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\texture_indirect_functions.h"
template< class T> static typename __nv_itex_trait< T> ::type 
#line 388
tex2DLod(T *ptr, ::cudaTextureObject_t obj, float x, float y, float level) {int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)level;::exit(___);}
#if 0
#line 389
{ 
#line 393
} 
#endif
#line 395 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 396
tex2DLod(::cudaTextureObject_t texObject, float x, float y, float level) {int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)level;::exit(___);}
#if 0
#line 397
{ 
#line 403
} 
#endif
#line 407 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\texture_indirect_functions.h"
template< class T> static typename __nv_itex_trait< T> ::type 
#line 408
tex2DLod(T *ptr, ::cudaTextureObject_t obj, float x, float y, float level, bool *isResident) {int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)level;(void)isResident;::exit(___);}
#if 0
#line 409
{ 
#line 415
} 
#endif
#line 417 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 418
tex2DLod(::cudaTextureObject_t texObject, float x, float y, float level, bool *isResident) {int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)level;(void)isResident;::exit(___);}
#if 0
#line 419
{ 
#line 425
} 
#endif
#line 430 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\texture_indirect_functions.h"
template< class T> static typename __nv_itex_trait< T> ::type 
#line 431
tex3DLod(T *ptr, ::cudaTextureObject_t obj, float x, float y, float z, float level) {int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;(void)level;::exit(___);}
#if 0
#line 432
{ 
#line 436
} 
#endif
#line 438 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 439
tex3DLod(::cudaTextureObject_t texObject, float x, float y, float z, float level) {int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;(void)level;::exit(___);}
#if 0
#line 440
{ 
#line 446
} 
#endif
#line 449 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\texture_indirect_functions.h"
template< class T> static typename __nv_itex_trait< T> ::type 
#line 450
tex3DLod(T *ptr, ::cudaTextureObject_t obj, float x, float y, float z, float level, bool *isResident) {int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;(void)level;(void)isResident;::exit(___);}
#if 0
#line 451
{ 
#line 457
} 
#endif
#line 459 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 460
tex3DLod(::cudaTextureObject_t texObject, float x, float y, float z, float level, bool *isResident) {int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;(void)level;(void)isResident;::exit(___);}
#if 0
#line 461
{ 
#line 467
} 
#endif
#line 472 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\texture_indirect_functions.h"
template< class T> static typename __nv_itex_trait< T> ::type 
#line 473
tex1DLayeredLod(T *ptr, ::cudaTextureObject_t obj, float x, int layer, float level) {int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)layer;(void)level;::exit(___);}
#if 0
#line 474
{ 
#line 478
} 
#endif
#line 480 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 481
tex1DLayeredLod(::cudaTextureObject_t texObject, float x, int layer, float level) {int volatile ___ = 1;(void)texObject;(void)x;(void)layer;(void)level;::exit(___);}
#if 0
#line 482
{ 
#line 488
} 
#endif
#line 491 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\texture_indirect_functions.h"
template< class T> static typename __nv_itex_trait< T> ::type 
#line 492
tex2DLayeredLod(T *ptr, ::cudaTextureObject_t obj, float x, float y, int layer, float level) {int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)layer;(void)level;::exit(___);}
#if 0
#line 493
{ 
#line 497
} 
#endif
#line 499 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 500
tex2DLayeredLod(::cudaTextureObject_t texObject, float x, float y, int layer, float level) {int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)layer;(void)level;::exit(___);}
#if 0
#line 501
{ 
#line 507
} 
#endif
#line 510 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\texture_indirect_functions.h"
template< class T> static typename __nv_itex_trait< T> ::type 
#line 511
tex2DLayeredLod(T *ptr, ::cudaTextureObject_t obj, float x, float y, int layer, float level, bool *isResident) {int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)layer;(void)level;(void)isResident;::exit(___);}
#if 0
#line 512
{ 
#line 518
} 
#endif
#line 520 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 521
tex2DLayeredLod(::cudaTextureObject_t texObject, float x, float y, int layer, float level, bool *isResident) {int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)layer;(void)level;(void)isResident;::exit(___);}
#if 0
#line 522
{ 
#line 528
} 
#endif
#line 531 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\texture_indirect_functions.h"
template< class T> static typename __nv_itex_trait< T> ::type 
#line 532
texCubemapLod(T *ptr, ::cudaTextureObject_t obj, float x, float y, float z, float level) {int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;(void)level;::exit(___);}
#if 0
#line 533
{ 
#line 537
} 
#endif
#line 539 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 540
texCubemapLod(::cudaTextureObject_t texObject, float x, float y, float z, float level) {int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;(void)level;::exit(___);}
#if 0
#line 541
{ 
#line 547
} 
#endif
#line 550 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\texture_indirect_functions.h"
template< class T> static typename __nv_itex_trait< T> ::type 
#line 551
texCubemapGrad(T *ptr, ::cudaTextureObject_t obj, float x, float y, float z, ::float4 dPdx, ::float4 dPdy) {int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;(void)dPdx;(void)dPdy;::exit(___);}
#if 0
#line 552
{ 
#line 556
} 
#endif
#line 558 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 559
texCubemapGrad(::cudaTextureObject_t texObject, float x, float y, float z, ::float4 dPdx, ::float4 dPdy) {int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;(void)dPdx;(void)dPdy;::exit(___);}
#if 0
#line 560
{ 
#line 566
} 
#endif
#line 568 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\texture_indirect_functions.h"
template< class T> static typename __nv_itex_trait< T> ::type 
#line 569
texCubemapLayeredLod(T *ptr, ::cudaTextureObject_t obj, float x, float y, float z, int layer, float level) {int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;(void)layer;(void)level;::exit(___);}
#if 0
#line 570
{ 
#line 574
} 
#endif
#line 576 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 577
texCubemapLayeredLod(::cudaTextureObject_t texObject, float x, float y, float z, int layer, float level) {int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;(void)layer;(void)level;::exit(___);}
#if 0
#line 578
{ 
#line 584
} 
#endif
#line 586 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\texture_indirect_functions.h"
template< class T> static typename __nv_itex_trait< T> ::type 
#line 587
tex1DGrad(T *ptr, ::cudaTextureObject_t obj, float x, float dPdx, float dPdy) {int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)dPdx;(void)dPdy;::exit(___);}
#if 0
#line 588
{ 
#line 592
} 
#endif
#line 594 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 595
tex1DGrad(::cudaTextureObject_t texObject, float x, float dPdx, float dPdy) {int volatile ___ = 1;(void)texObject;(void)x;(void)dPdx;(void)dPdy;::exit(___);}
#if 0
#line 596
{ 
#line 602
} 
#endif
#line 605 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\texture_indirect_functions.h"
template< class T> static typename __nv_itex_trait< T> ::type 
#line 606
tex2DGrad(T *ptr, ::cudaTextureObject_t obj, float x, float y, ::float2 dPdx, ::float2 dPdy) {int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)dPdx;(void)dPdy;::exit(___);}
#if 0
#line 607
{ 
#line 612
} 
#endif
#line 614 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 615
tex2DGrad(::cudaTextureObject_t texObject, float x, float y, ::float2 dPdx, ::float2 dPdy) {int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)dPdx;(void)dPdy;::exit(___);}
#if 0
#line 616
{ 
#line 622
} 
#endif
#line 625 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\texture_indirect_functions.h"
template< class T> static typename __nv_itex_trait< T> ::type 
#line 626
tex2DGrad(T *ptr, ::cudaTextureObject_t obj, float x, float y, ::float2 dPdx, ::float2 dPdy, bool *isResident) {int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)dPdx;(void)dPdy;(void)isResident;::exit(___);}
#if 0
#line 627
{ 
#line 634
} 
#endif
#line 636 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 637
tex2DGrad(::cudaTextureObject_t texObject, float x, float y, ::float2 dPdx, ::float2 dPdy, bool *isResident) {int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)dPdx;(void)dPdy;(void)isResident;::exit(___);}
#if 0
#line 638
{ 
#line 644
} 
#endif
#line 648 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\texture_indirect_functions.h"
template< class T> static typename __nv_itex_trait< T> ::type 
#line 649
tex3DGrad(T *ptr, ::cudaTextureObject_t obj, float x, float y, float z, ::float4 dPdx, ::float4 dPdy) {int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;(void)dPdx;(void)dPdy;::exit(___);}
#if 0
#line 650
{ 
#line 654
} 
#endif
#line 656 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 657
tex3DGrad(::cudaTextureObject_t texObject, float x, float y, float z, ::float4 dPdx, ::float4 dPdy) {int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;(void)dPdx;(void)dPdy;::exit(___);}
#if 0
#line 658
{ 
#line 664
} 
#endif
#line 667 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\texture_indirect_functions.h"
template< class T> static typename __nv_itex_trait< T> ::type 
#line 668
tex3DGrad(T *ptr, ::cudaTextureObject_t obj, float x, float y, float z, ::float4 dPdx, ::float4 dPdy, bool *isResident) {int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;(void)dPdx;(void)dPdy;(void)isResident;::exit(___);}
#if 0
#line 669
{ 
#line 675
} 
#endif
#line 677 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 678
tex3DGrad(::cudaTextureObject_t texObject, float x, float y, float z, ::float4 dPdx, ::float4 dPdy, bool *isResident) {int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;(void)dPdx;(void)dPdy;(void)isResident;::exit(___);}
#if 0
#line 679
{ 
#line 685
} 
#endif
#line 690 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\texture_indirect_functions.h"
template< class T> static typename __nv_itex_trait< T> ::type 
#line 691
tex1DLayeredGrad(T *ptr, ::cudaTextureObject_t obj, float x, int layer, float dPdx, float dPdy) {int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)layer;(void)dPdx;(void)dPdy;::exit(___);}
#if 0
#line 692
{ 
#line 696
} 
#endif
#line 698 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 699
tex1DLayeredGrad(::cudaTextureObject_t texObject, float x, int layer, float dPdx, float dPdy) {int volatile ___ = 1;(void)texObject;(void)x;(void)layer;(void)dPdx;(void)dPdy;::exit(___);}
#if 0
#line 700
{ 
#line 706
} 
#endif
#line 709 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\texture_indirect_functions.h"
template< class T> static typename __nv_itex_trait< T> ::type 
#line 710
tex2DLayeredGrad(T *ptr, ::cudaTextureObject_t obj, float x, float y, int layer, ::float2 dPdx, ::float2 dPdy) {int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)layer;(void)dPdx;(void)dPdy;::exit(___);}
#if 0
#line 711
{ 
#line 715
} 
#endif
#line 717 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 718
tex2DLayeredGrad(::cudaTextureObject_t texObject, float x, float y, int layer, ::float2 dPdx, ::float2 dPdy) {int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)layer;(void)dPdx;(void)dPdy;::exit(___);}
#if 0
#line 719
{ 
#line 725
} 
#endif
#line 728 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\texture_indirect_functions.h"
template< class T> static typename __nv_itex_trait< T> ::type 
#line 729
tex2DLayeredGrad(T *ptr, ::cudaTextureObject_t obj, float x, float y, int layer, ::float2 dPdx, ::float2 dPdy, bool *isResident) {int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)layer;(void)dPdx;(void)dPdy;(void)isResident;::exit(___);}
#if 0
#line 730
{ 
#line 736
} 
#endif
#line 738 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 739
tex2DLayeredGrad(::cudaTextureObject_t texObject, float x, float y, int layer, ::float2 dPdx, ::float2 dPdy, bool *isResident) {int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)layer;(void)dPdx;(void)dPdy;(void)isResident;::exit(___);}
#if 0
#line 740
{ 
#line 746
} 
#endif
#line 750 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\texture_indirect_functions.h"
template< class T> static typename __nv_itex_trait< T> ::type 
#line 751
texCubemapLayeredGrad(T *ptr, ::cudaTextureObject_t obj, float x, float y, float z, int layer, ::float4 dPdx, ::float4 dPdy) {int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;(void)layer;(void)dPdx;(void)dPdy;::exit(___);}
#if 0
#line 752
{ 
#line 756
} 
#endif
#line 758 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 759
texCubemapLayeredGrad(::cudaTextureObject_t texObject, float x, float y, float z, int layer, ::float4 dPdx, ::float4 dPdy) {int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;(void)layer;(void)dPdx;(void)dPdy;::exit(___);}
#if 0
#line 760
{ 
#line 766
} 
#endif
#line 59 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\surface_indirect_functions.h"
template< class T> struct __nv_isurf_trait { }; 
#line 60
template<> struct __nv_isurf_trait< char>  { typedef void type; }; 
#line 61
template<> struct __nv_isurf_trait< signed char>  { typedef void type; }; 
#line 62
template<> struct __nv_isurf_trait< char1>  { typedef void type; }; 
#line 63
template<> struct __nv_isurf_trait< unsigned char>  { typedef void type; }; 
#line 64
template<> struct __nv_isurf_trait< uchar1>  { typedef void type; }; 
#line 65
template<> struct __nv_isurf_trait< short>  { typedef void type; }; 
#line 66
template<> struct __nv_isurf_trait< short1>  { typedef void type; }; 
#line 67
template<> struct __nv_isurf_trait< unsigned short>  { typedef void type; }; 
#line 68
template<> struct __nv_isurf_trait< ushort1>  { typedef void type; }; 
#line 69
template<> struct __nv_isurf_trait< int>  { typedef void type; }; 
#line 70
template<> struct __nv_isurf_trait< int1>  { typedef void type; }; 
#line 71
template<> struct __nv_isurf_trait< unsigned>  { typedef void type; }; 
#line 72
template<> struct __nv_isurf_trait< uint1>  { typedef void type; }; 
#line 73
template<> struct __nv_isurf_trait< __int64>  { typedef void type; }; 
#line 74
template<> struct __nv_isurf_trait< longlong1>  { typedef void type; }; 
#line 75
template<> struct __nv_isurf_trait< unsigned __int64>  { typedef void type; }; 
#line 76
template<> struct __nv_isurf_trait< ulonglong1>  { typedef void type; }; 
#line 77
template<> struct __nv_isurf_trait< float>  { typedef void type; }; 
#line 78
template<> struct __nv_isurf_trait< float1>  { typedef void type; }; 
#line 80
template<> struct __nv_isurf_trait< char2>  { typedef void type; }; 
#line 81
template<> struct __nv_isurf_trait< uchar2>  { typedef void type; }; 
#line 82
template<> struct __nv_isurf_trait< short2>  { typedef void type; }; 
#line 83
template<> struct __nv_isurf_trait< ushort2>  { typedef void type; }; 
#line 84
template<> struct __nv_isurf_trait< int2>  { typedef void type; }; 
#line 85
template<> struct __nv_isurf_trait< uint2>  { typedef void type; }; 
#line 86
template<> struct __nv_isurf_trait< longlong2>  { typedef void type; }; 
#line 87
template<> struct __nv_isurf_trait< ulonglong2>  { typedef void type; }; 
#line 88
template<> struct __nv_isurf_trait< float2>  { typedef void type; }; 
#line 90
template<> struct __nv_isurf_trait< char4>  { typedef void type; }; 
#line 91
template<> struct __nv_isurf_trait< uchar4>  { typedef void type; }; 
#line 92
template<> struct __nv_isurf_trait< short4>  { typedef void type; }; 
#line 93
template<> struct __nv_isurf_trait< ushort4>  { typedef void type; }; 
#line 94
template<> struct __nv_isurf_trait< int4>  { typedef void type; }; 
#line 95
template<> struct __nv_isurf_trait< uint4>  { typedef void type; }; 
#line 96
template<> struct __nv_isurf_trait< float4>  { typedef void type; }; 
#line 99
template< class T> static typename __nv_isurf_trait< T> ::type 
#line 100
surf1Dread(T *ptr, ::cudaSurfaceObject_t obj, int x, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)mode;::exit(___);}
#if 0
#line 101
{ 
#line 105
} 
#endif
#line 107 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\surface_indirect_functions.h"
template< class T> static T 
#line 108
surf1Dread(::cudaSurfaceObject_t surfObject, int x, ::cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)surfObject;(void)x;(void)boundaryMode;::exit(___);}
#if 0
#line 109
{ 
#line 115
} 
#endif
#line 117 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\surface_indirect_functions.h"
template< class T> static typename __nv_isurf_trait< T> ::type 
#line 118
surf2Dread(T *ptr, ::cudaSurfaceObject_t obj, int x, int y, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)mode;::exit(___);}
#if 0
#line 119
{ 
#line 123
} 
#endif
#line 125 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\surface_indirect_functions.h"
template< class T> static T 
#line 126
surf2Dread(::cudaSurfaceObject_t surfObject, int x, int y, ::cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)surfObject;(void)x;(void)y;(void)boundaryMode;::exit(___);}
#if 0
#line 127
{ 
#line 133
} 
#endif
#line 136 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\surface_indirect_functions.h"
template< class T> static typename __nv_isurf_trait< T> ::type 
#line 137
surf3Dread(T *ptr, ::cudaSurfaceObject_t obj, int x, int y, int z, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;(void)mode;::exit(___);}
#if 0
#line 138
{ 
#line 142
} 
#endif
#line 144 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\surface_indirect_functions.h"
template< class T> static T 
#line 145
surf3Dread(::cudaSurfaceObject_t surfObject, int x, int y, int z, ::cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)surfObject;(void)x;(void)y;(void)z;(void)boundaryMode;::exit(___);}
#if 0
#line 146
{ 
#line 152
} 
#endif
#line 154 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\surface_indirect_functions.h"
template< class T> static typename __nv_isurf_trait< T> ::type 
#line 155
surf1DLayeredread(T *ptr, ::cudaSurfaceObject_t obj, int x, int layer, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)layer;(void)mode;::exit(___);}
#if 0
#line 156
{ 
#line 160
} 
#endif
#line 162 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\surface_indirect_functions.h"
template< class T> static T 
#line 163
surf1DLayeredread(::cudaSurfaceObject_t surfObject, int x, int layer, ::cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)surfObject;(void)x;(void)layer;(void)boundaryMode;::exit(___);}
#if 0
#line 164
{ 
#line 170
} 
#endif
#line 172 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\surface_indirect_functions.h"
template< class T> static typename __nv_isurf_trait< T> ::type 
#line 173
surf2DLayeredread(T *ptr, ::cudaSurfaceObject_t obj, int x, int y, int layer, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)layer;(void)mode;::exit(___);}
#if 0
#line 174
{ 
#line 178
} 
#endif
#line 180 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\surface_indirect_functions.h"
template< class T> static T 
#line 181
surf2DLayeredread(::cudaSurfaceObject_t surfObject, int x, int y, int layer, ::cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)surfObject;(void)x;(void)y;(void)layer;(void)boundaryMode;::exit(___);}
#if 0
#line 182
{ 
#line 188
} 
#endif
#line 190 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\surface_indirect_functions.h"
template< class T> static typename __nv_isurf_trait< T> ::type 
#line 191
surfCubemapread(T *ptr, ::cudaSurfaceObject_t obj, int x, int y, int face, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)face;(void)mode;::exit(___);}
#if 0
#line 192
{ 
#line 196
} 
#endif
#line 198 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\surface_indirect_functions.h"
template< class T> static T 
#line 199
surfCubemapread(::cudaSurfaceObject_t surfObject, int x, int y, int face, ::cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)surfObject;(void)x;(void)y;(void)face;(void)boundaryMode;::exit(___);}
#if 0
#line 200
{ 
#line 206
} 
#endif
#line 208 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\surface_indirect_functions.h"
template< class T> static typename __nv_isurf_trait< T> ::type 
#line 209
surfCubemapLayeredread(T *ptr, ::cudaSurfaceObject_t obj, int x, int y, int layerface, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)layerface;(void)mode;::exit(___);}
#if 0
#line 210
{ 
#line 214
} 
#endif
#line 216 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\surface_indirect_functions.h"
template< class T> static T 
#line 217
surfCubemapLayeredread(::cudaSurfaceObject_t surfObject, int x, int y, int layerface, ::cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)surfObject;(void)x;(void)y;(void)layerface;(void)boundaryMode;::exit(___);}
#if 0
#line 218
{ 
#line 224
} 
#endif
#line 226 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\surface_indirect_functions.h"
template< class T> static typename __nv_isurf_trait< T> ::type 
#line 227
surf1Dwrite(T val, ::cudaSurfaceObject_t obj, int x, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)val;(void)obj;(void)x;(void)mode;::exit(___);}
#if 0
#line 228
{ 
#line 232
} 
#endif
#line 234 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\surface_indirect_functions.h"
template< class T> static typename __nv_isurf_trait< T> ::type 
#line 235
surf2Dwrite(T val, ::cudaSurfaceObject_t obj, int x, int y, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)val;(void)obj;(void)x;(void)y;(void)mode;::exit(___);}
#if 0
#line 236
{ 
#line 240
} 
#endif
#line 242 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\surface_indirect_functions.h"
template< class T> static typename __nv_isurf_trait< T> ::type 
#line 243
surf3Dwrite(T val, ::cudaSurfaceObject_t obj, int x, int y, int z, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)val;(void)obj;(void)x;(void)y;(void)z;(void)mode;::exit(___);}
#if 0
#line 244
{ 
#line 248
} 
#endif
#line 250 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\surface_indirect_functions.h"
template< class T> static typename __nv_isurf_trait< T> ::type 
#line 251
surf1DLayeredwrite(T val, ::cudaSurfaceObject_t obj, int x, int layer, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)val;(void)obj;(void)x;(void)layer;(void)mode;::exit(___);}
#if 0
#line 252
{ 
#line 256
} 
#endif
#line 258 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\surface_indirect_functions.h"
template< class T> static typename __nv_isurf_trait< T> ::type 
#line 259
surf2DLayeredwrite(T val, ::cudaSurfaceObject_t obj, int x, int y, int layer, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)val;(void)obj;(void)x;(void)y;(void)layer;(void)mode;::exit(___);}
#if 0
#line 260
{ 
#line 264
} 
#endif
#line 266 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\surface_indirect_functions.h"
template< class T> static typename __nv_isurf_trait< T> ::type 
#line 267
surfCubemapwrite(T val, ::cudaSurfaceObject_t obj, int x, int y, int face, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)val;(void)obj;(void)x;(void)y;(void)face;(void)mode;::exit(___);}
#if 0
#line 268
{ 
#line 272
} 
#endif
#line 274 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\surface_indirect_functions.h"
template< class T> static typename __nv_isurf_trait< T> ::type 
#line 275
surfCubemapLayeredwrite(T val, ::cudaSurfaceObject_t obj, int x, int y, int layerface, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)val;(void)obj;(void)x;(void)y;(void)layerface;(void)mode;::exit(___);}
#if 0
#line 276
{ 
#line 280
} 
#endif
#line 3309 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\crt/device_functions.h"
extern "C" unsigned __stdcall __cudaPushCallConfiguration(dim3 gridDim, dim3 blockDim, size_t sharedMem = 0, CUstream_st * stream = 0); 
#line 68 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\device_launch_parameters.h"
extern "C" {
#line 71 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\device_launch_parameters.h"
extern const uint3 __device_builtin_variable_threadIdx; 
#line 72
extern const uint3 __device_builtin_variable_blockIdx; 
#line 73
extern const dim3 __device_builtin_variable_blockDim; 
#line 74
extern const dim3 __device_builtin_variable_gridDim; 
#line 75
extern const int __device_builtin_variable_warpSize; 
#line 80
}
#line 205 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/bin/../include\\cuda_runtime.h"
template< class T> static __inline ::cudaError_t 
#line 206
cudaLaunchKernel(const T *
#line 207
func, ::dim3 
#line 208
gridDim, ::dim3 
#line 209
blockDim, void **
#line 210
args, ::size_t 
#line 211
sharedMem = 0, ::cudaStream_t 
#line 212
stream = 0) 
#line 214
{ 
#line 215
return ::cudaLaunchKernel((const void *)func, gridDim, blockDim, args, sharedMem, stream); 
#line 216
} 
#line 339 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/bin/../include\\cuda_runtime.h"
template< class T> static __inline ::cudaError_t 
#line 340
cudaLaunchCooperativeKernel(const T *
#line 341
func, ::dim3 
#line 342
gridDim, ::dim3 
#line 343
blockDim, void **
#line 344
args, ::size_t 
#line 345
sharedMem = 0, ::cudaStream_t 
#line 346
stream = 0) 
#line 348
{ 
#line 349
return ::cudaLaunchCooperativeKernel((const void *)func, gridDim, blockDim, args, sharedMem, stream); 
#line 350
} 
#line 383
static __inline cudaError_t cudaEventCreate(cudaEvent_t *
#line 384
event, unsigned 
#line 385
flags) 
#line 387
{ 
#line 388
return ::cudaEventCreateWithFlags(event, flags); 
#line 389
} 
#line 448
static __inline cudaError_t cudaMallocHost(void **
#line 449
ptr, size_t 
#line 450
size, unsigned 
#line 451
flags) 
#line 453
{ 
#line 454
return ::cudaHostAlloc(ptr, size, flags); 
#line 455
} 
#line 457
template< class T> static __inline ::cudaError_t 
#line 458
cudaHostAlloc(T **
#line 459
ptr, ::size_t 
#line 460
size, unsigned 
#line 461
flags) 
#line 463
{ 
#line 464
return ::cudaHostAlloc((void **)((void *)ptr), size, flags); 
#line 465
} 
#line 467
template< class T> static __inline ::cudaError_t 
#line 468
cudaHostGetDevicePointer(T **
#line 469
pDevice, void *
#line 470
pHost, unsigned 
#line 471
flags) 
#line 473
{ 
#line 474
return ::cudaHostGetDevicePointer((void **)((void *)pDevice), pHost, flags); 
#line 475
} 
#line 577
template< class T> static __inline ::cudaError_t 
#line 578
cudaMallocManaged(T **
#line 579
devPtr, ::size_t 
#line 580
size, unsigned 
#line 581
flags = 1) 
#line 583
{ 
#line 584
return ::cudaMallocManaged((void **)((void *)devPtr), size, flags); 
#line 585
} 
#line 667
template< class T> static __inline ::cudaError_t 
#line 668
cudaStreamAttachMemAsync(::cudaStream_t 
#line 669
stream, T *
#line 670
devPtr, ::size_t 
#line 671
length = 0, unsigned 
#line 672
flags = 4) 
#line 674
{ 
#line 675
return ::cudaStreamAttachMemAsync(stream, (void *)devPtr, length, flags); 
#line 676
} 
#line 678
template< class T> __inline ::cudaError_t 
#line 679
cudaMalloc(T **
#line 680
devPtr, ::size_t 
#line 681
size) 
#line 683
{ 
#line 684
return ::cudaMalloc((void **)((void *)devPtr), size); 
#line 685
} 
#line 687
template< class T> static __inline ::cudaError_t 
#line 688
cudaMallocHost(T **
#line 689
ptr, ::size_t 
#line 690
size, unsigned 
#line 691
flags = 0) 
#line 693
{ 
#line 694
return cudaMallocHost((void **)((void *)ptr), size, flags); 
#line 695
} 
#line 697
template< class T> static __inline ::cudaError_t 
#line 698
cudaMallocPitch(T **
#line 699
devPtr, ::size_t *
#line 700
pitch, ::size_t 
#line 701
width, ::size_t 
#line 702
height) 
#line 704
{ 
#line 705
return ::cudaMallocPitch((void **)((void *)devPtr), pitch, width, height); 
#line 706
} 
#line 717
static __inline cudaError_t cudaMallocAsync(void **
#line 718
ptr, size_t 
#line 719
size, cudaMemPool_t 
#line 720
memPool, cudaStream_t 
#line 721
stream) 
#line 723
{ 
#line 724
return ::cudaMallocFromPoolAsync(ptr, size, memPool, stream); 
#line 725
} 
#line 727
template< class T> static __inline ::cudaError_t 
#line 728
cudaMallocAsync(T **
#line 729
ptr, ::size_t 
#line 730
size, ::cudaMemPool_t 
#line 731
memPool, ::cudaStream_t 
#line 732
stream) 
#line 734
{ 
#line 735
return ::cudaMallocFromPoolAsync((void **)((void *)ptr), size, memPool, stream); 
#line 736
} 
#line 738
template< class T> static __inline ::cudaError_t 
#line 739
cudaMallocAsync(T **
#line 740
ptr, ::size_t 
#line 741
size, ::cudaStream_t 
#line 742
stream) 
#line 744
{ 
#line 745
return ::cudaMallocAsync((void **)((void *)ptr), size, stream); 
#line 746
} 
#line 748
template< class T> static __inline ::cudaError_t 
#line 749
cudaMallocFromPoolAsync(T **
#line 750
ptr, ::size_t 
#line 751
size, ::cudaMemPool_t 
#line 752
memPool, ::cudaStream_t 
#line 753
stream) 
#line 755
{ 
#line 756
return ::cudaMallocFromPoolAsync((void **)((void *)ptr), size, memPool, stream); 
#line 757
} 
#line 796
template< class T> static __inline ::cudaError_t 
#line 797
cudaMemcpyToSymbol(const T &
#line 798
symbol, const void *
#line 799
src, ::size_t 
#line 800
count, ::size_t 
#line 801
offset = 0, ::cudaMemcpyKind 
#line 802
kind = cudaMemcpyHostToDevice) 
#line 804
{ 
#line 805
return ::cudaMemcpyToSymbol((const void *)(&symbol), src, count, offset, kind); 
#line 806
} 
#line 850
template< class T> static __inline ::cudaError_t 
#line 851
cudaMemcpyToSymbolAsync(const T &
#line 852
symbol, const void *
#line 853
src, ::size_t 
#line 854
count, ::size_t 
#line 855
offset = 0, ::cudaMemcpyKind 
#line 856
kind = cudaMemcpyHostToDevice, ::cudaStream_t 
#line 857
stream = 0) 
#line 859
{ 
#line 860
return ::cudaMemcpyToSymbolAsync((const void *)(&symbol), src, count, offset, kind, stream); 
#line 861
} 
#line 898
template< class T> static __inline ::cudaError_t 
#line 899
cudaMemcpyFromSymbol(void *
#line 900
dst, const T &
#line 901
symbol, ::size_t 
#line 902
count, ::size_t 
#line 903
offset = 0, ::cudaMemcpyKind 
#line 904
kind = cudaMemcpyDeviceToHost) 
#line 906
{ 
#line 907
return ::cudaMemcpyFromSymbol(dst, (const void *)(&symbol), count, offset, kind); 
#line 908
} 
#line 952
template< class T> static __inline ::cudaError_t 
#line 953
cudaMemcpyFromSymbolAsync(void *
#line 954
dst, const T &
#line 955
symbol, ::size_t 
#line 956
count, ::size_t 
#line 957
offset = 0, ::cudaMemcpyKind 
#line 958
kind = cudaMemcpyDeviceToHost, ::cudaStream_t 
#line 959
stream = 0) 
#line 961
{ 
#line 962
return ::cudaMemcpyFromSymbolAsync(dst, (const void *)(&symbol), count, offset, kind, stream); 
#line 963
} 
#line 1021
template< class T> static __inline ::cudaError_t 
#line 1022
cudaGraphAddMemcpyNodeToSymbol(::cudaGraphNode_t *
#line 1023
pGraphNode, ::cudaGraph_t 
#line 1024
graph, const ::cudaGraphNode_t *
#line 1025
pDependencies, ::size_t 
#line 1026
numDependencies, const T &
#line 1027
symbol, const void *
#line 1028
src, ::size_t 
#line 1029
count, ::size_t 
#line 1030
offset, ::cudaMemcpyKind 
#line 1031
kind) 
#line 1032
{ 
#line 1033
return ::cudaGraphAddMemcpyNodeToSymbol(pGraphNode, graph, pDependencies, numDependencies, (const void *)(&symbol), src, count, offset, kind); 
#line 1034
} 
#line 1092
template< class T> static __inline ::cudaError_t 
#line 1093
cudaGraphAddMemcpyNodeFromSymbol(::cudaGraphNode_t *
#line 1094
pGraphNode, ::cudaGraph_t 
#line 1095
graph, const ::cudaGraphNode_t *
#line 1096
pDependencies, ::size_t 
#line 1097
numDependencies, void *
#line 1098
dst, const T &
#line 1099
symbol, ::size_t 
#line 1100
count, ::size_t 
#line 1101
offset, ::cudaMemcpyKind 
#line 1102
kind) 
#line 1103
{ 
#line 1104
return ::cudaGraphAddMemcpyNodeFromSymbol(pGraphNode, graph, pDependencies, numDependencies, dst, (const void *)(&symbol), count, offset, kind); 
#line 1105
} 
#line 1143
template< class T> static __inline ::cudaError_t 
#line 1144
cudaGraphMemcpyNodeSetParamsToSymbol(::cudaGraphNode_t 
#line 1145
node, const T &
#line 1146
symbol, const void *
#line 1147
src, ::size_t 
#line 1148
count, ::size_t 
#line 1149
offset, ::cudaMemcpyKind 
#line 1150
kind) 
#line 1151
{ 
#line 1152
return ::cudaGraphMemcpyNodeSetParamsToSymbol(node, (const void *)(&symbol), src, count, offset, kind); 
#line 1153
} 
#line 1191
template< class T> static __inline ::cudaError_t 
#line 1192
cudaGraphMemcpyNodeSetParamsFromSymbol(::cudaGraphNode_t 
#line 1193
node, void *
#line 1194
dst, const T &
#line 1195
symbol, ::size_t 
#line 1196
count, ::size_t 
#line 1197
offset, ::cudaMemcpyKind 
#line 1198
kind) 
#line 1199
{ 
#line 1200
return ::cudaGraphMemcpyNodeSetParamsFromSymbol(node, dst, (const void *)(&symbol), count, offset, kind); 
#line 1201
} 
#line 1249
template< class T> static __inline ::cudaError_t 
#line 1250
cudaGraphExecMemcpyNodeSetParamsToSymbol(::cudaGraphExec_t 
#line 1251
hGraphExec, ::cudaGraphNode_t 
#line 1252
node, const T &
#line 1253
symbol, const void *
#line 1254
src, ::size_t 
#line 1255
count, ::size_t 
#line 1256
offset, ::cudaMemcpyKind 
#line 1257
kind) 
#line 1258
{ 
#line 1259
return ::cudaGraphExecMemcpyNodeSetParamsToSymbol(hGraphExec, node, (const void *)(&symbol), src, count, offset, kind); 
#line 1260
} 
#line 1308
template< class T> static __inline ::cudaError_t 
#line 1309
cudaGraphExecMemcpyNodeSetParamsFromSymbol(::cudaGraphExec_t 
#line 1310
hGraphExec, ::cudaGraphNode_t 
#line 1311
node, void *
#line 1312
dst, const T &
#line 1313
symbol, ::size_t 
#line 1314
count, ::size_t 
#line 1315
offset, ::cudaMemcpyKind 
#line 1316
kind) 
#line 1317
{ 
#line 1318
return ::cudaGraphExecMemcpyNodeSetParamsFromSymbol(hGraphExec, node, dst, (const void *)(&symbol), count, offset, kind); 
#line 1319
} 
#line 1397 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/bin/../include\\cuda_runtime.h"
template< class T> static __inline ::cudaError_t 
#line 1398
cudaGetSymbolAddress(void **
#line 1399
devPtr, const T &
#line 1400
symbol) 
#line 1402
{ 
#line 1403
return ::cudaGetSymbolAddress(devPtr, (const void *)(&symbol)); 
#line 1404
} 
#line 1429
template< class T> static __inline ::cudaError_t 
#line 1430
cudaGetSymbolSize(::size_t *
#line 1431
size, const T &
#line 1432
symbol) 
#line 1434
{ 
#line 1435
return ::cudaGetSymbolSize(size, (const void *)(&symbol)); 
#line 1436
} 
#line 1473
template< class T, int dim, cudaTextureReadMode readMode> 
#line 1474
__declspec(deprecated) static __inline ::cudaError_t cudaBindTexture(::size_t *
#line 1475
offset, const texture< T, dim, readMode>  &
#line 1476
tex, const void *
#line 1477
devPtr, const ::cudaChannelFormatDesc &
#line 1478
desc, ::size_t 
#line 1479
size = 4294967295U) 
#line 1481
{ 
#line 1482
return ::cudaBindTexture(offset, &tex, devPtr, &desc, size); 
#line 1483
} 
#line 1519
template< class T, int dim, cudaTextureReadMode readMode> 
#line 1520
__declspec(deprecated) static __inline ::cudaError_t cudaBindTexture(::size_t *
#line 1521
offset, const texture< T, dim, readMode>  &
#line 1522
tex, const void *
#line 1523
devPtr, ::size_t 
#line 1524
size = 4294967295U) 
#line 1526
{ 
#line 1527
return cudaBindTexture(offset, tex, devPtr, (tex.channelDesc), size); 
#line 1528
} 
#line 1576
template< class T, int dim, cudaTextureReadMode readMode> 
#line 1577
__declspec(deprecated) static __inline ::cudaError_t cudaBindTexture2D(::size_t *
#line 1578
offset, const texture< T, dim, readMode>  &
#line 1579
tex, const void *
#line 1580
devPtr, const ::cudaChannelFormatDesc &
#line 1581
desc, ::size_t 
#line 1582
width, ::size_t 
#line 1583
height, ::size_t 
#line 1584
pitch) 
#line 1586
{ 
#line 1587
return ::cudaBindTexture2D(offset, &tex, devPtr, &desc, width, height, pitch); 
#line 1588
} 
#line 1635
template< class T, int dim, cudaTextureReadMode readMode> 
#line 1636
__declspec(deprecated) static __inline ::cudaError_t cudaBindTexture2D(::size_t *
#line 1637
offset, const texture< T, dim, readMode>  &
#line 1638
tex, const void *
#line 1639
devPtr, ::size_t 
#line 1640
width, ::size_t 
#line 1641
height, ::size_t 
#line 1642
pitch) 
#line 1644
{ 
#line 1645
return ::cudaBindTexture2D(offset, &tex, devPtr, &(tex.channelDesc), width, height, pitch); 
#line 1646
} 
#line 1678
template< class T, int dim, cudaTextureReadMode readMode> 
#line 1679
__declspec(deprecated) static __inline ::cudaError_t cudaBindTextureToArray(const texture< T, dim, readMode>  &
#line 1680
tex, ::cudaArray_const_t 
#line 1681
array, const ::cudaChannelFormatDesc &
#line 1682
desc) 
#line 1684
{ 
#line 1685
return ::cudaBindTextureToArray(&tex, array, &desc); 
#line 1686
} 
#line 1717
template< class T, int dim, cudaTextureReadMode readMode> 
#line 1718
__declspec(deprecated) static __inline ::cudaError_t cudaBindTextureToArray(const texture< T, dim, readMode>  &
#line 1719
tex, ::cudaArray_const_t 
#line 1720
array) 
#line 1722
{ 
#line 1723
::cudaChannelFormatDesc desc; 
#line 1724
::cudaError_t err = ::cudaGetChannelDesc(&desc, array); 
#line 1726
return (err == (cudaSuccess)) ? cudaBindTextureToArray(tex, array, desc) : err; 
#line 1727
} 
#line 1759
template< class T, int dim, cudaTextureReadMode readMode> 
#line 1760
__declspec(deprecated) static __inline ::cudaError_t cudaBindTextureToMipmappedArray(const texture< T, dim, readMode>  &
#line 1761
tex, ::cudaMipmappedArray_const_t 
#line 1762
mipmappedArray, const ::cudaChannelFormatDesc &
#line 1763
desc) 
#line 1765
{ 
#line 1766
return ::cudaBindTextureToMipmappedArray(&tex, mipmappedArray, &desc); 
#line 1767
} 
#line 1798
template< class T, int dim, cudaTextureReadMode readMode> 
#line 1799
__declspec(deprecated) static __inline ::cudaError_t cudaBindTextureToMipmappedArray(const texture< T, dim, readMode>  &
#line 1800
tex, ::cudaMipmappedArray_const_t 
#line 1801
mipmappedArray) 
#line 1803
{ 
#line 1804
::cudaChannelFormatDesc desc; 
#line 1805
::cudaArray_t levelArray; 
#line 1806
::cudaError_t err = ::cudaGetMipmappedArrayLevel(&levelArray, mipmappedArray, 0); 
#line 1808
if (err != (cudaSuccess)) { 
#line 1809
return err; 
#line 1810
}  
#line 1811
err = ::cudaGetChannelDesc(&desc, levelArray); 
#line 1813
return (err == (cudaSuccess)) ? cudaBindTextureToMipmappedArray(tex, mipmappedArray, desc) : err; 
#line 1814
} 
#line 1841
template< class T, int dim, cudaTextureReadMode readMode> 
#line 1842
__declspec(deprecated) static __inline ::cudaError_t cudaUnbindTexture(const texture< T, dim, readMode>  &
#line 1843
tex) 
#line 1845
{ 
#line 1846
return ::cudaUnbindTexture(&tex); 
#line 1847
} 
#line 1877
template< class T, int dim, cudaTextureReadMode readMode> 
#line 1878
__declspec(deprecated) static __inline ::cudaError_t cudaGetTextureAlignmentOffset(::size_t *
#line 1879
offset, const texture< T, dim, readMode>  &
#line 1880
tex) 
#line 1882
{ 
#line 1883
return ::cudaGetTextureAlignmentOffset(offset, &tex); 
#line 1884
} 
#line 1929
template< class T> static __inline ::cudaError_t 
#line 1930
cudaFuncSetCacheConfig(T *
#line 1931
func, ::cudaFuncCache 
#line 1932
cacheConfig) 
#line 1934
{ 
#line 1935
return ::cudaFuncSetCacheConfig((const void *)func, cacheConfig); 
#line 1936
} 
#line 1938
template< class T> static __inline ::cudaError_t 
#line 1939
cudaFuncSetSharedMemConfig(T *
#line 1940
func, ::cudaSharedMemConfig 
#line 1941
config) 
#line 1943
{ 
#line 1944
return ::cudaFuncSetSharedMemConfig((const void *)func, config); 
#line 1945
} 
#line 1977 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/bin/../include\\cuda_runtime.h"
template< class T> __inline ::cudaError_t 
#line 1978
cudaOccupancyMaxActiveBlocksPerMultiprocessor(int *
#line 1979
numBlocks, T 
#line 1980
func, int 
#line 1981
blockSize, ::size_t 
#line 1982
dynamicSMemSize) 
#line 1983
{ 
#line 1984
return ::cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(numBlocks, (const void *)func, blockSize, dynamicSMemSize, 0); 
#line 1985
} 
#line 2029
template< class T> __inline ::cudaError_t 
#line 2030
cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int *
#line 2031
numBlocks, T 
#line 2032
func, int 
#line 2033
blockSize, ::size_t 
#line 2034
dynamicSMemSize, unsigned 
#line 2035
flags) 
#line 2036
{ 
#line 2037
return ::cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(numBlocks, (const void *)func, blockSize, dynamicSMemSize, flags); 
#line 2038
} 
#line 2043
class __cudaOccupancyB2DHelper { 
#line 2044
size_t n; 
#line 2046
public: __cudaOccupancyB2DHelper(size_t n_) : n(n_) { } 
#line 2047
size_t operator()(int) 
#line 2048
{ 
#line 2049
return n; 
#line 2050
} 
#line 2051
}; 
#line 2099
template< class UnaryFunction, class T> static __inline ::cudaError_t 
#line 2100
cudaOccupancyMaxPotentialBlockSizeVariableSMemWithFlags(int *
#line 2101
minGridSize, int *
#line 2102
blockSize, T 
#line 2103
func, UnaryFunction 
#line 2104
blockSizeToDynamicSMemSize, int 
#line 2105
blockSizeLimit = 0, unsigned 
#line 2106
flags = 0) 
#line 2107
{ 
#line 2108
::cudaError_t status; 
#line 2111
int device; 
#line 2112
::cudaFuncAttributes attr; 
#line 2115
int maxThreadsPerMultiProcessor; 
#line 2116
int warpSize; 
#line 2117
int devMaxThreadsPerBlock; 
#line 2118
int multiProcessorCount; 
#line 2119
int funcMaxThreadsPerBlock; 
#line 2120
int occupancyLimit; 
#line 2121
int granularity; 
#line 2124
int maxBlockSize = 0; 
#line 2125
int numBlocks = 0; 
#line 2126
int maxOccupancy = 0; 
#line 2129
int blockSizeToTryAligned; 
#line 2130
int blockSizeToTry; 
#line 2131
int blockSizeLimitAligned; 
#line 2132
int occupancyInBlocks; 
#line 2133
int occupancyInThreads; 
#line 2134
::size_t dynamicSMemSize; 
#line 2140
if (((!minGridSize) || (!blockSize)) || (!func)) { 
#line 2141
return cudaErrorInvalidValue; 
#line 2142
}  
#line 2148
status = ::cudaGetDevice(&device); 
#line 2149
if (status != (cudaSuccess)) { 
#line 2150
return status; 
#line 2151
}  
#line 2153
status = cudaDeviceGetAttribute(&maxThreadsPerMultiProcessor, cudaDevAttrMaxThreadsPerMultiProcessor, device); 
#line 2157
if (status != (cudaSuccess)) { 
#line 2158
return status; 
#line 2159
}  
#line 2161
status = cudaDeviceGetAttribute(&warpSize, cudaDevAttrWarpSize, device); 
#line 2165
if (status != (cudaSuccess)) { 
#line 2166
return status; 
#line 2167
}  
#line 2169
status = cudaDeviceGetAttribute(&devMaxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock, device); 
#line 2173
if (status != (cudaSuccess)) { 
#line 2174
return status; 
#line 2175
}  
#line 2177
status = cudaDeviceGetAttribute(&multiProcessorCount, cudaDevAttrMultiProcessorCount, device); 
#line 2181
if (status != (cudaSuccess)) { 
#line 2182
return status; 
#line 2183
}  
#line 2185
status = cudaFuncGetAttributes(&attr, func); 
#line 2186
if (status != (cudaSuccess)) { 
#line 2187
return status; 
#line 2188
}  
#line 2190
funcMaxThreadsPerBlock = (attr.maxThreadsPerBlock); 
#line 2196
occupancyLimit = maxThreadsPerMultiProcessor; 
#line 2197
granularity = warpSize; 
#line 2199
if (blockSizeLimit == 0) { 
#line 2200
blockSizeLimit = devMaxThreadsPerBlock; 
#line 2201
}  
#line 2203
if (devMaxThreadsPerBlock < blockSizeLimit) { 
#line 2204
blockSizeLimit = devMaxThreadsPerBlock; 
#line 2205
}  
#line 2207
if (funcMaxThreadsPerBlock < blockSizeLimit) { 
#line 2208
blockSizeLimit = funcMaxThreadsPerBlock; 
#line 2209
}  
#line 2211
blockSizeLimitAligned = (((blockSizeLimit + (granularity - 1)) / granularity) * granularity); 
#line 2213
for (blockSizeToTryAligned = blockSizeLimitAligned; blockSizeToTryAligned > 0; blockSizeToTryAligned -= granularity) { 
#line 2217
if (blockSizeLimit < blockSizeToTryAligned) { 
#line 2218
blockSizeToTry = blockSizeLimit; 
#line 2219
} else { 
#line 2220
blockSizeToTry = blockSizeToTryAligned; 
#line 2221
}  
#line 2223
dynamicSMemSize = blockSizeToDynamicSMemSize(blockSizeToTry); 
#line 2225
status = cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(&occupancyInBlocks, func, blockSizeToTry, dynamicSMemSize, flags); 
#line 2232
if (status != (cudaSuccess)) { 
#line 2233
return status; 
#line 2234
}  
#line 2236
occupancyInThreads = (blockSizeToTry * occupancyInBlocks); 
#line 2238
if (occupancyInThreads > maxOccupancy) { 
#line 2239
maxBlockSize = blockSizeToTry; 
#line 2240
numBlocks = occupancyInBlocks; 
#line 2241
maxOccupancy = occupancyInThreads; 
#line 2242
}  
#line 2246
if (occupancyLimit == maxOccupancy) { 
#line 2247
break; 
#line 2248
}  
#line 2249
}  
#line 2257
(*minGridSize) = (numBlocks * multiProcessorCount); 
#line 2258
(*blockSize) = maxBlockSize; 
#line 2260
return status; 
#line 2261
} 
#line 2295
template< class UnaryFunction, class T> static __inline ::cudaError_t 
#line 2296
cudaOccupancyMaxPotentialBlockSizeVariableSMem(int *
#line 2297
minGridSize, int *
#line 2298
blockSize, T 
#line 2299
func, UnaryFunction 
#line 2300
blockSizeToDynamicSMemSize, int 
#line 2301
blockSizeLimit = 0) 
#line 2302
{ 
#line 2303
return cudaOccupancyMaxPotentialBlockSizeVariableSMemWithFlags(minGridSize, blockSize, func, blockSizeToDynamicSMemSize, blockSizeLimit, 0); 
#line 2304
} 
#line 2341
template< class T> static __inline ::cudaError_t 
#line 2342
cudaOccupancyMaxPotentialBlockSize(int *
#line 2343
minGridSize, int *
#line 2344
blockSize, T 
#line 2345
func, ::size_t 
#line 2346
dynamicSMemSize = 0, int 
#line 2347
blockSizeLimit = 0) 
#line 2348
{ 
#line 2349
return cudaOccupancyMaxPotentialBlockSizeVariableSMemWithFlags(minGridSize, blockSize, func, ((::__cudaOccupancyB2DHelper)(dynamicSMemSize)), blockSizeLimit, 0); 
#line 2350
} 
#line 2379
template< class T> static __inline ::cudaError_t 
#line 2380
cudaOccupancyAvailableDynamicSMemPerBlock(::size_t *
#line 2381
dynamicSmemSize, T 
#line 2382
func, int 
#line 2383
numBlocks, int 
#line 2384
blockSize) 
#line 2385
{ 
#line 2386
return ::cudaOccupancyAvailableDynamicSMemPerBlock(dynamicSmemSize, (const void *)func, numBlocks, blockSize); 
#line 2387
} 
#line 2438
template< class T> static __inline ::cudaError_t 
#line 2439
cudaOccupancyMaxPotentialBlockSizeWithFlags(int *
#line 2440
minGridSize, int *
#line 2441
blockSize, T 
#line 2442
func, ::size_t 
#line 2443
dynamicSMemSize = 0, int 
#line 2444
blockSizeLimit = 0, unsigned 
#line 2445
flags = 0) 
#line 2446
{ 
#line 2447
return cudaOccupancyMaxPotentialBlockSizeVariableSMemWithFlags(minGridSize, blockSize, func, ((::__cudaOccupancyB2DHelper)(dynamicSMemSize)), blockSizeLimit, flags); 
#line 2448
} 
#line 2482
template< class T> static __inline ::cudaError_t 
#line 2483
cudaOccupancyMaxPotentialClusterSize(int *
#line 2484
clusterSize, T *
#line 2485
func, const ::cudaLaunchConfig_t *
#line 2486
config) 
#line 2487
{ 
#line 2488
return ::cudaOccupancyMaxPotentialClusterSize(clusterSize, (const void *)func, config); 
#line 2489
} 
#line 2525
template< class T> static __inline ::cudaError_t 
#line 2526
cudaOccupancyMaxActiveClusters(int *
#line 2527
numClusters, T *
#line 2528
func, const ::cudaLaunchConfig_t *
#line 2529
config) 
#line 2530
{ 
#line 2531
return ::cudaOccupancyMaxActiveClusters(numClusters, (const void *)func, config); 
#line 2532
} 
#line 2565
template< class T> __inline ::cudaError_t 
#line 2566
cudaFuncGetAttributes(::cudaFuncAttributes *
#line 2567
attr, T *
#line 2568
entry) 
#line 2570
{ 
#line 2571
return ::cudaFuncGetAttributes(attr, (const void *)entry); 
#line 2572
} 
#line 2627
template< class T> static __inline ::cudaError_t 
#line 2628
cudaFuncSetAttribute(T *
#line 2629
entry, ::cudaFuncAttribute 
#line 2630
attr, int 
#line 2631
value) 
#line 2633
{ 
#line 2634
return ::cudaFuncSetAttribute((const void *)entry, attr, value); 
#line 2635
} 
#line 2659
template< class T, int dim> 
#line 2660
__declspec(deprecated) static __inline ::cudaError_t cudaBindSurfaceToArray(const surface< T, dim>  &
#line 2661
surf, ::cudaArray_const_t 
#line 2662
array, const ::cudaChannelFormatDesc &
#line 2663
desc) 
#line 2665
{ 
#line 2666
return ::cudaBindSurfaceToArray(&surf, array, &desc); 
#line 2667
} 
#line 2690
template< class T, int dim> 
#line 2691
__declspec(deprecated) static __inline ::cudaError_t cudaBindSurfaceToArray(const surface< T, dim>  &
#line 2692
surf, ::cudaArray_const_t 
#line 2693
array) 
#line 2695
{ 
#line 2696
::cudaChannelFormatDesc desc; 
#line 2697
::cudaError_t err = ::cudaGetChannelDesc(&desc, array); 
#line 2699
return (err == (cudaSuccess)) ? cudaBindSurfaceToArray(surf, array, desc) : err; 
#line 2700
} 
#line 2714 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/bin/../include\\cuda_runtime.h"
#pragma warning(pop)
#line 64 "CMakeCUDACompilerId.cu"
const char *info_compiler = ("INFO:compiler[NVIDIA]"); 
#line 66
const char *info_simulate = ("INFO:simulate[MSVC]"); 
#line 369 "CMakeCUDACompilerId.cu"
const char info_version[] = {'I', 'N', 'F', 'O', ':', 'c', 'o', 'm', 'p', 'i', 'l', 'e', 'r', '_', 'v', 'e', 'r', 's', 'i', 'o', 'n', '[', (('0') + ((11 / 10000000) % 10)), (('0') + ((11 / 1000000) % 10)), (('0') + ((11 / 100000) % 10)), (('0') + ((11 / 10000) % 10)), (('0') + ((11 / 1000) % 10)), (('0') + ((11 / 100) % 10)), (('0') + ((11 / 10) % 10)), (('0') + (11 % 10)), '.', (('0') + ((8 / 10000000) % 10)), (('0') + ((8 / 1000000) % 10)), (('0') + ((8 / 100000) % 10)), (('0') + ((8 / 10000) % 10)), (('0') + ((8 / 1000) % 10)), (('0') + ((8 / 100) % 10)), (('0') + ((8 / 10) % 10)), (('0') + (8 % 10)), '.', (('0') + ((89 / 10000000) % 10)), (('0') + ((89 / 1000000) % 10)), (('0') + ((89 / 100000) % 10)), (('0') + ((89 / 10000) % 10)), (('0') + ((89 / 1000) % 10)), (('0') + ((89 / 100) % 10)), (('0') + ((89 / 10) % 10)), (('0') + (89 % 10)), ']', '\000'}; 
#line 398 "CMakeCUDACompilerId.cu"
const char info_simulate_version[] = {'I', 'N', 'F', 'O', ':', 's', 'i', 'm', 'u', 'l', 'a', 't', 'e', '_', 'v', 'e', 'r', 's', 'i', 'o', 'n', '[', (('0') + (((1937 / 100) / 10000000) % 10)), (('0') + (((1937 / 100) / 1000000) % 10)), (('0') + (((1937 / 100) / 100000) % 10)), (('0') + (((1937 / 100) / 10000) % 10)), (('0') + (((1937 / 100) / 1000) % 10)), (('0') + (((1937 / 100) / 100) % 10)), (('0') + (((1937 / 100) / 10) % 10)), (('0') + ((1937 / 100) % 10)), '.', (('0') + (((1937 % 100) / 10000000) % 10)), (('0') + (((1937 % 100) / 1000000) % 10)), (('0') + (((1937 % 100) / 100000) % 10)), (('0') + (((1937 % 100) / 10000) % 10)), (('0') + (((1937 % 100) / 1000) % 10)), (('0') + (((1937 % 100) / 100) % 10)), (('0') + (((1937 % 100) / 10) % 10)), (('0') + ((1937 % 100) % 10)), ']', '\000'}; 
#line 418 "CMakeCUDACompilerId.cu"
const char *info_platform = ("INFO:platform[Windows]"); 
#line 419
const char *info_arch = ("INFO:arch[x64]"); 
#line 423
const char *info_language_standard_default = ("INFO:standard_default[03]"); 
#line 439 "CMakeCUDACompilerId.cu"
const char *info_language_extensions_default = ("INFO:extensions_default[OFF]"); 
#line 450 "CMakeCUDACompilerId.cu"
int main(int argc, char *argv[]) 
#line 451
{ 
#line 452
int require = 0; 
#line 453
require += (info_compiler[argc]); 
#line 454
require += (info_platform[argc]); 
#line 456
require += (info_version[argc]); 
#line 459 "CMakeCUDACompilerId.cu"
require += (info_simulate[argc]); 
#line 462 "CMakeCUDACompilerId.cu"
require += (info_simulate_version[argc]); 
#line 464 "CMakeCUDACompilerId.cu"
require += (info_language_standard_default[argc]); 
#line 465
require += (info_language_extensions_default[argc]); 
#line 466
(void)argv; 
#line 467
return require; 
#line 468
} 
#line 1 "CMakeCUDACompilerId.cudafe1.stub.c"
#define _NV_ANON_NAMESPACE _GLOBAL__N__be01f2cb_22_CMakeCUDACompilerId_cu_bd57c623
#ifdef _NV_ANON_NAMESPACE
#endif
#pragma pack()
#line 1 "CMakeCUDACompilerId.cudafe1.stub.c"
#include "CMakeCUDACompilerId.cudafe1.stub.c"
#line 1 "CMakeCUDACompilerId.cudafe1.stub.c"
#undef _NV_ANON_NAMESPACE
