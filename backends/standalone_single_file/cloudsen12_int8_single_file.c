/* Auto-generated standalone INT8 single-file build unit */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <limits.h>
#include <math.h>
#include <omp.h>

/* BEGIN INLINE v9_plan.h */
#include <stdint.h>
typedef enum {OP_CONV,OP_RELU6,OP_RELU,OP_ADD,OP_RESIZE,OP_CONCAT} OpKind;
typedef struct {int kind,out,a,b,layer; int32_t m1,s1,m2,s2; double r1,r2; int n,c,h,w;} Op;
typedef struct {int oc,icg,kh,kw,sh,sw,ph,pw,dh,dw,groups; const char *wf,*bf,*mf,*sf;} Layer;
extern const Op OPS[]; extern const int NOPS,NTENSORS; extern const Layer LAYERS[]; extern const int NLAYERS,FINAL_TENSOR;
/* END INLINE v9_plan.h */

/* BEGIN INLINE v9_plan.c */
const Layer LAYERS[] = {
{32,13,3,3,2,2,1,1,1,1,1,"weights_int8/conv_000_encoder.features.0.0.bin","biases_int32/conv_000_encoder.features.0.0.bin","fixed_point/conv_000_encoder.features.0.0_multipliers.bin","fixed_point/conv_000_encoder.features.0.0_shifts.bin"},
{32,1,3,3,1,1,1,1,1,1,32,"weights_int8/conv_001_encoder.features.1.conv.0.0.bin","biases_int32/conv_001_encoder.features.1.conv.0.0.bin","fixed_point/conv_001_encoder.features.1.conv.0.0_multipliers.bin","fixed_point/conv_001_encoder.features.1.conv.0.0_shifts.bin"},
{16,32,1,1,1,1,0,0,1,1,1,"weights_int8/conv_002_encoder.features.1.conv.1.bin","biases_int32/conv_002_encoder.features.1.conv.1.bin","fixed_point/conv_002_encoder.features.1.conv.1_multipliers.bin","fixed_point/conv_002_encoder.features.1.conv.1_shifts.bin"},
{96,16,1,1,1,1,0,0,1,1,1,"weights_int8/conv_003_encoder.features.2.conv.0.0.bin","biases_int32/conv_003_encoder.features.2.conv.0.0.bin","fixed_point/conv_003_encoder.features.2.conv.0.0_multipliers.bin","fixed_point/conv_003_encoder.features.2.conv.0.0_shifts.bin"},
{96,1,3,3,2,2,1,1,1,1,96,"weights_int8/conv_004_encoder.features.2.conv.1.0.bin","biases_int32/conv_004_encoder.features.2.conv.1.0.bin","fixed_point/conv_004_encoder.features.2.conv.1.0_multipliers.bin","fixed_point/conv_004_encoder.features.2.conv.1.0_shifts.bin"},
{24,96,1,1,1,1,0,0,1,1,1,"weights_int8/conv_005_encoder.features.2.conv.2.bin","biases_int32/conv_005_encoder.features.2.conv.2.bin","fixed_point/conv_005_encoder.features.2.conv.2_multipliers.bin","fixed_point/conv_005_encoder.features.2.conv.2_shifts.bin"},
{144,24,1,1,1,1,0,0,1,1,1,"weights_int8/conv_006_encoder.features.3.conv.0.0.bin","biases_int32/conv_006_encoder.features.3.conv.0.0.bin","fixed_point/conv_006_encoder.features.3.conv.0.0_multipliers.bin","fixed_point/conv_006_encoder.features.3.conv.0.0_shifts.bin"},
{144,1,3,3,1,1,1,1,1,1,144,"weights_int8/conv_007_encoder.features.3.conv.1.0.bin","biases_int32/conv_007_encoder.features.3.conv.1.0.bin","fixed_point/conv_007_encoder.features.3.conv.1.0_multipliers.bin","fixed_point/conv_007_encoder.features.3.conv.1.0_shifts.bin"},
{24,144,1,1,1,1,0,0,1,1,1,"weights_int8/conv_008_encoder.features.3.conv.2.bin","biases_int32/conv_008_encoder.features.3.conv.2.bin","fixed_point/conv_008_encoder.features.3.conv.2_multipliers.bin","fixed_point/conv_008_encoder.features.3.conv.2_shifts.bin"},
{144,24,1,1,1,1,0,0,1,1,1,"weights_int8/conv_009_encoder.features.4.conv.0.0.bin","biases_int32/conv_009_encoder.features.4.conv.0.0.bin","fixed_point/conv_009_encoder.features.4.conv.0.0_multipliers.bin","fixed_point/conv_009_encoder.features.4.conv.0.0_shifts.bin"},
{144,1,3,3,2,2,1,1,1,1,144,"weights_int8/conv_010_encoder.features.4.conv.1.0.bin","biases_int32/conv_010_encoder.features.4.conv.1.0.bin","fixed_point/conv_010_encoder.features.4.conv.1.0_multipliers.bin","fixed_point/conv_010_encoder.features.4.conv.1.0_shifts.bin"},
{32,144,1,1,1,1,0,0,1,1,1,"weights_int8/conv_011_encoder.features.4.conv.2.bin","biases_int32/conv_011_encoder.features.4.conv.2.bin","fixed_point/conv_011_encoder.features.4.conv.2_multipliers.bin","fixed_point/conv_011_encoder.features.4.conv.2_shifts.bin"},
{192,32,1,1,1,1,0,0,1,1,1,"weights_int8/conv_012_encoder.features.5.conv.0.0.bin","biases_int32/conv_012_encoder.features.5.conv.0.0.bin","fixed_point/conv_012_encoder.features.5.conv.0.0_multipliers.bin","fixed_point/conv_012_encoder.features.5.conv.0.0_shifts.bin"},
{192,1,3,3,1,1,1,1,1,1,192,"weights_int8/conv_013_encoder.features.5.conv.1.0.bin","biases_int32/conv_013_encoder.features.5.conv.1.0.bin","fixed_point/conv_013_encoder.features.5.conv.1.0_multipliers.bin","fixed_point/conv_013_encoder.features.5.conv.1.0_shifts.bin"},
{32,192,1,1,1,1,0,0,1,1,1,"weights_int8/conv_014_encoder.features.5.conv.2.bin","biases_int32/conv_014_encoder.features.5.conv.2.bin","fixed_point/conv_014_encoder.features.5.conv.2_multipliers.bin","fixed_point/conv_014_encoder.features.5.conv.2_shifts.bin"},
{192,32,1,1,1,1,0,0,1,1,1,"weights_int8/conv_015_encoder.features.6.conv.0.0.bin","biases_int32/conv_015_encoder.features.6.conv.0.0.bin","fixed_point/conv_015_encoder.features.6.conv.0.0_multipliers.bin","fixed_point/conv_015_encoder.features.6.conv.0.0_shifts.bin"},
{192,1,3,3,1,1,1,1,1,1,192,"weights_int8/conv_016_encoder.features.6.conv.1.0.bin","biases_int32/conv_016_encoder.features.6.conv.1.0.bin","fixed_point/conv_016_encoder.features.6.conv.1.0_multipliers.bin","fixed_point/conv_016_encoder.features.6.conv.1.0_shifts.bin"},
{32,192,1,1,1,1,0,0,1,1,1,"weights_int8/conv_017_encoder.features.6.conv.2.bin","biases_int32/conv_017_encoder.features.6.conv.2.bin","fixed_point/conv_017_encoder.features.6.conv.2_multipliers.bin","fixed_point/conv_017_encoder.features.6.conv.2_shifts.bin"},
{192,32,1,1,1,1,0,0,1,1,1,"weights_int8/conv_018_encoder.features.7.conv.0.0.bin","biases_int32/conv_018_encoder.features.7.conv.0.0.bin","fixed_point/conv_018_encoder.features.7.conv.0.0_multipliers.bin","fixed_point/conv_018_encoder.features.7.conv.0.0_shifts.bin"},
{192,1,3,3,2,2,1,1,1,1,192,"weights_int8/conv_019_encoder.features.7.conv.1.0.bin","biases_int32/conv_019_encoder.features.7.conv.1.0.bin","fixed_point/conv_019_encoder.features.7.conv.1.0_multipliers.bin","fixed_point/conv_019_encoder.features.7.conv.1.0_shifts.bin"},
{64,192,1,1,1,1,0,0,1,1,1,"weights_int8/conv_020_encoder.features.7.conv.2.bin","biases_int32/conv_020_encoder.features.7.conv.2.bin","fixed_point/conv_020_encoder.features.7.conv.2_multipliers.bin","fixed_point/conv_020_encoder.features.7.conv.2_shifts.bin"},
{384,64,1,1,1,1,0,0,1,1,1,"weights_int8/conv_021_encoder.features.8.conv.0.0.bin","biases_int32/conv_021_encoder.features.8.conv.0.0.bin","fixed_point/conv_021_encoder.features.8.conv.0.0_multipliers.bin","fixed_point/conv_021_encoder.features.8.conv.0.0_shifts.bin"},
{384,1,3,3,1,1,1,1,1,1,384,"weights_int8/conv_022_encoder.features.8.conv.1.0.bin","biases_int32/conv_022_encoder.features.8.conv.1.0.bin","fixed_point/conv_022_encoder.features.8.conv.1.0_multipliers.bin","fixed_point/conv_022_encoder.features.8.conv.1.0_shifts.bin"},
{64,384,1,1,1,1,0,0,1,1,1,"weights_int8/conv_023_encoder.features.8.conv.2.bin","biases_int32/conv_023_encoder.features.8.conv.2.bin","fixed_point/conv_023_encoder.features.8.conv.2_multipliers.bin","fixed_point/conv_023_encoder.features.8.conv.2_shifts.bin"},
{384,64,1,1,1,1,0,0,1,1,1,"weights_int8/conv_024_encoder.features.9.conv.0.0.bin","biases_int32/conv_024_encoder.features.9.conv.0.0.bin","fixed_point/conv_024_encoder.features.9.conv.0.0_multipliers.bin","fixed_point/conv_024_encoder.features.9.conv.0.0_shifts.bin"},
{384,1,3,3,1,1,1,1,1,1,384,"weights_int8/conv_025_encoder.features.9.conv.1.0.bin","biases_int32/conv_025_encoder.features.9.conv.1.0.bin","fixed_point/conv_025_encoder.features.9.conv.1.0_multipliers.bin","fixed_point/conv_025_encoder.features.9.conv.1.0_shifts.bin"},
{64,384,1,1,1,1,0,0,1,1,1,"weights_int8/conv_026_encoder.features.9.conv.2.bin","biases_int32/conv_026_encoder.features.9.conv.2.bin","fixed_point/conv_026_encoder.features.9.conv.2_multipliers.bin","fixed_point/conv_026_encoder.features.9.conv.2_shifts.bin"},
{384,64,1,1,1,1,0,0,1,1,1,"weights_int8/conv_027_encoder.features.10.conv.0.0.bin","biases_int32/conv_027_encoder.features.10.conv.0.0.bin","fixed_point/conv_027_encoder.features.10.conv.0.0_multipliers.bin","fixed_point/conv_027_encoder.features.10.conv.0.0_shifts.bin"},
{384,1,3,3,1,1,1,1,1,1,384,"weights_int8/conv_028_encoder.features.10.conv.1.0.bin","biases_int32/conv_028_encoder.features.10.conv.1.0.bin","fixed_point/conv_028_encoder.features.10.conv.1.0_multipliers.bin","fixed_point/conv_028_encoder.features.10.conv.1.0_shifts.bin"},
{64,384,1,1,1,1,0,0,1,1,1,"weights_int8/conv_029_encoder.features.10.conv.2.bin","biases_int32/conv_029_encoder.features.10.conv.2.bin","fixed_point/conv_029_encoder.features.10.conv.2_multipliers.bin","fixed_point/conv_029_encoder.features.10.conv.2_shifts.bin"},
{384,64,1,1,1,1,0,0,1,1,1,"weights_int8/conv_030_encoder.features.11.conv.0.0.bin","biases_int32/conv_030_encoder.features.11.conv.0.0.bin","fixed_point/conv_030_encoder.features.11.conv.0.0_multipliers.bin","fixed_point/conv_030_encoder.features.11.conv.0.0_shifts.bin"},
{384,1,3,3,1,1,1,1,1,1,384,"weights_int8/conv_031_encoder.features.11.conv.1.0.bin","biases_int32/conv_031_encoder.features.11.conv.1.0.bin","fixed_point/conv_031_encoder.features.11.conv.1.0_multipliers.bin","fixed_point/conv_031_encoder.features.11.conv.1.0_shifts.bin"},
{96,384,1,1,1,1,0,0,1,1,1,"weights_int8/conv_032_encoder.features.11.conv.2.bin","biases_int32/conv_032_encoder.features.11.conv.2.bin","fixed_point/conv_032_encoder.features.11.conv.2_multipliers.bin","fixed_point/conv_032_encoder.features.11.conv.2_shifts.bin"},
{576,96,1,1,1,1,0,0,1,1,1,"weights_int8/conv_033_encoder.features.12.conv.0.0.bin","biases_int32/conv_033_encoder.features.12.conv.0.0.bin","fixed_point/conv_033_encoder.features.12.conv.0.0_multipliers.bin","fixed_point/conv_033_encoder.features.12.conv.0.0_shifts.bin"},
{576,1,3,3,1,1,1,1,1,1,576,"weights_int8/conv_034_encoder.features.12.conv.1.0.bin","biases_int32/conv_034_encoder.features.12.conv.1.0.bin","fixed_point/conv_034_encoder.features.12.conv.1.0_multipliers.bin","fixed_point/conv_034_encoder.features.12.conv.1.0_shifts.bin"},
{96,576,1,1,1,1,0,0,1,1,1,"weights_int8/conv_035_encoder.features.12.conv.2.bin","biases_int32/conv_035_encoder.features.12.conv.2.bin","fixed_point/conv_035_encoder.features.12.conv.2_multipliers.bin","fixed_point/conv_035_encoder.features.12.conv.2_shifts.bin"},
{576,96,1,1,1,1,0,0,1,1,1,"weights_int8/conv_036_encoder.features.13.conv.0.0.bin","biases_int32/conv_036_encoder.features.13.conv.0.0.bin","fixed_point/conv_036_encoder.features.13.conv.0.0_multipliers.bin","fixed_point/conv_036_encoder.features.13.conv.0.0_shifts.bin"},
{576,1,3,3,1,1,1,1,1,1,576,"weights_int8/conv_037_encoder.features.13.conv.1.0.bin","biases_int32/conv_037_encoder.features.13.conv.1.0.bin","fixed_point/conv_037_encoder.features.13.conv.1.0_multipliers.bin","fixed_point/conv_037_encoder.features.13.conv.1.0_shifts.bin"},
{96,576,1,1,1,1,0,0,1,1,1,"weights_int8/conv_038_encoder.features.13.conv.2.bin","biases_int32/conv_038_encoder.features.13.conv.2.bin","fixed_point/conv_038_encoder.features.13.conv.2_multipliers.bin","fixed_point/conv_038_encoder.features.13.conv.2_shifts.bin"},
{576,96,1,1,1,1,0,0,1,1,1,"weights_int8/conv_039_encoder.features.14.conv.0.0.bin","biases_int32/conv_039_encoder.features.14.conv.0.0.bin","fixed_point/conv_039_encoder.features.14.conv.0.0_multipliers.bin","fixed_point/conv_039_encoder.features.14.conv.0.0_shifts.bin"},
{576,1,3,3,2,2,1,1,1,1,576,"weights_int8/conv_040_encoder.features.14.conv.1.0.bin","biases_int32/conv_040_encoder.features.14.conv.1.0.bin","fixed_point/conv_040_encoder.features.14.conv.1.0_multipliers.bin","fixed_point/conv_040_encoder.features.14.conv.1.0_shifts.bin"},
{160,576,1,1,1,1,0,0,1,1,1,"weights_int8/conv_041_encoder.features.14.conv.2.bin","biases_int32/conv_041_encoder.features.14.conv.2.bin","fixed_point/conv_041_encoder.features.14.conv.2_multipliers.bin","fixed_point/conv_041_encoder.features.14.conv.2_shifts.bin"},
{960,160,1,1,1,1,0,0,1,1,1,"weights_int8/conv_042_encoder.features.15.conv.0.0.bin","biases_int32/conv_042_encoder.features.15.conv.0.0.bin","fixed_point/conv_042_encoder.features.15.conv.0.0_multipliers.bin","fixed_point/conv_042_encoder.features.15.conv.0.0_shifts.bin"},
{960,1,3,3,1,1,1,1,1,1,960,"weights_int8/conv_043_encoder.features.15.conv.1.0.bin","biases_int32/conv_043_encoder.features.15.conv.1.0.bin","fixed_point/conv_043_encoder.features.15.conv.1.0_multipliers.bin","fixed_point/conv_043_encoder.features.15.conv.1.0_shifts.bin"},
{160,960,1,1,1,1,0,0,1,1,1,"weights_int8/conv_044_encoder.features.15.conv.2.bin","biases_int32/conv_044_encoder.features.15.conv.2.bin","fixed_point/conv_044_encoder.features.15.conv.2_multipliers.bin","fixed_point/conv_044_encoder.features.15.conv.2_shifts.bin"},
{960,160,1,1,1,1,0,0,1,1,1,"weights_int8/conv_045_encoder.features.16.conv.0.0.bin","biases_int32/conv_045_encoder.features.16.conv.0.0.bin","fixed_point/conv_045_encoder.features.16.conv.0.0_multipliers.bin","fixed_point/conv_045_encoder.features.16.conv.0.0_shifts.bin"},
{960,1,3,3,1,1,1,1,1,1,960,"weights_int8/conv_046_encoder.features.16.conv.1.0.bin","biases_int32/conv_046_encoder.features.16.conv.1.0.bin","fixed_point/conv_046_encoder.features.16.conv.1.0_multipliers.bin","fixed_point/conv_046_encoder.features.16.conv.1.0_shifts.bin"},
{160,960,1,1,1,1,0,0,1,1,1,"weights_int8/conv_047_encoder.features.16.conv.2.bin","biases_int32/conv_047_encoder.features.16.conv.2.bin","fixed_point/conv_047_encoder.features.16.conv.2_multipliers.bin","fixed_point/conv_047_encoder.features.16.conv.2_shifts.bin"},
{960,160,1,1,1,1,0,0,1,1,1,"weights_int8/conv_048_encoder.features.17.conv.0.0.bin","biases_int32/conv_048_encoder.features.17.conv.0.0.bin","fixed_point/conv_048_encoder.features.17.conv.0.0_multipliers.bin","fixed_point/conv_048_encoder.features.17.conv.0.0_shifts.bin"},
{960,1,3,3,1,1,1,1,1,1,960,"weights_int8/conv_049_encoder.features.17.conv.1.0.bin","biases_int32/conv_049_encoder.features.17.conv.1.0.bin","fixed_point/conv_049_encoder.features.17.conv.1.0_multipliers.bin","fixed_point/conv_049_encoder.features.17.conv.1.0_shifts.bin"},
{320,960,1,1,1,1,0,0,1,1,1,"weights_int8/conv_050_encoder.features.17.conv.2.bin","biases_int32/conv_050_encoder.features.17.conv.2.bin","fixed_point/conv_050_encoder.features.17.conv.2_multipliers.bin","fixed_point/conv_050_encoder.features.17.conv.2_shifts.bin"},
{1280,320,1,1,1,1,0,0,1,1,1,"weights_int8/conv_051_encoder.features.18.0.bin","biases_int32/conv_051_encoder.features.18.0.bin","fixed_point/conv_051_encoder.features.18.0_multipliers.bin","fixed_point/conv_051_encoder.features.18.0_shifts.bin"},
{256,1376,3,3,1,1,1,1,1,1,1,"weights_int8/conv_052_decoder.blocks.0.conv1.0.bin","biases_int32/conv_052_decoder.blocks.0.conv1.0.bin","fixed_point/conv_052_decoder.blocks.0.conv1.0_multipliers.bin","fixed_point/conv_052_decoder.blocks.0.conv1.0_shifts.bin"},
{256,256,3,3,1,1,1,1,1,1,1,"weights_int8/conv_053_decoder.blocks.0.conv2.0.bin","biases_int32/conv_053_decoder.blocks.0.conv2.0.bin","fixed_point/conv_053_decoder.blocks.0.conv2.0_multipliers.bin","fixed_point/conv_053_decoder.blocks.0.conv2.0_shifts.bin"},
{128,288,3,3,1,1,1,1,1,1,1,"weights_int8/conv_054_decoder.blocks.1.conv1.0.bin","biases_int32/conv_054_decoder.blocks.1.conv1.0.bin","fixed_point/conv_054_decoder.blocks.1.conv1.0_multipliers.bin","fixed_point/conv_054_decoder.blocks.1.conv1.0_shifts.bin"},
{128,128,3,3,1,1,1,1,1,1,1,"weights_int8/conv_055_decoder.blocks.1.conv2.0.bin","biases_int32/conv_055_decoder.blocks.1.conv2.0.bin","fixed_point/conv_055_decoder.blocks.1.conv2.0_multipliers.bin","fixed_point/conv_055_decoder.blocks.1.conv2.0_shifts.bin"},
{64,152,3,3,1,1,1,1,1,1,1,"weights_int8/conv_056_decoder.blocks.2.conv1.0.bin","biases_int32/conv_056_decoder.blocks.2.conv1.0.bin","fixed_point/conv_056_decoder.blocks.2.conv1.0_multipliers.bin","fixed_point/conv_056_decoder.blocks.2.conv1.0_shifts.bin"},
{64,64,3,3,1,1,1,1,1,1,1,"weights_int8/conv_057_decoder.blocks.2.conv2.0.bin","biases_int32/conv_057_decoder.blocks.2.conv2.0.bin","fixed_point/conv_057_decoder.blocks.2.conv2.0_multipliers.bin","fixed_point/conv_057_decoder.blocks.2.conv2.0_shifts.bin"},
{32,80,3,3,1,1,1,1,1,1,1,"weights_int8/conv_058_decoder.blocks.3.conv1.0.bin","biases_int32/conv_058_decoder.blocks.3.conv1.0.bin","fixed_point/conv_058_decoder.blocks.3.conv1.0_multipliers.bin","fixed_point/conv_058_decoder.blocks.3.conv1.0_shifts.bin"},
{32,32,3,3,1,1,1,1,1,1,1,"weights_int8/conv_059_decoder.blocks.3.conv2.0.bin","biases_int32/conv_059_decoder.blocks.3.conv2.0.bin","fixed_point/conv_059_decoder.blocks.3.conv2.0_multipliers.bin","fixed_point/conv_059_decoder.blocks.3.conv2.0_shifts.bin"},
{16,32,3,3,1,1,1,1,1,1,1,"weights_int8/conv_060_decoder.blocks.4.conv1.0.bin","biases_int32/conv_060_decoder.blocks.4.conv1.0.bin","fixed_point/conv_060_decoder.blocks.4.conv1.0_multipliers.bin","fixed_point/conv_060_decoder.blocks.4.conv1.0_shifts.bin"},
{16,16,3,3,1,1,1,1,1,1,1,"weights_int8/conv_061_decoder.blocks.4.conv2.0.bin","biases_int32/conv_061_decoder.blocks.4.conv2.0.bin","fixed_point/conv_061_decoder.blocks.4.conv2.0_multipliers.bin","fixed_point/conv_061_decoder.blocks.4.conv2.0_shifts.bin"},
{4,16,3,3,1,1,1,1,1,1,1,"weights_int8/conv_062_segmentation_head.0.bin","biases_int32/conv_062_segmentation_head.0.bin","fixed_point/conv_062_segmentation_head.0_multipliers.bin","fixed_point/conv_062_segmentation_head.0_shifts.bin"},
};
const Op OPS[] = {
{OP_CONV,1,0,-1,0,1073741824,1,0,0,1,1,1,32,128,128},
{OP_RELU6,2,1,-1,-1,0,0,143,0,1,1,1,32,128,128},
{OP_CONV,3,2,-1,1,1170688993,1,0,0,1.0902890866032806,1,1,32,128,128},
{OP_RELU6,4,3,-1,-1,0,0,15,0,1,1,1,32,128,128},
{OP_CONV,5,4,-1,2,1142879147,4,0,0,8.5151131947835292,1,1,16,128,128},
{OP_CONV,6,5,-1,3,1073741824,1,0,0,1,1,1,96,128,128},
{OP_RELU6,7,6,-1,-1,0,0,52,0,1,1,1,96,128,128},
{OP_CONV,8,7,-1,4,1301235541,2,0,0,2.4237400690714517,1,1,96,64,64},
{OP_RELU6,9,8,-1,-1,0,0,49,0,1,1,1,96,64,64},
{OP_CONV,10,9,-1,5,1442332809,2,0,0,2.6865542099596431,1,1,24,64,64},
{OP_CONV,11,10,-1,6,1073741824,1,0,0,1,1,1,144,64,64},
{OP_RELU6,12,11,-1,-1,0,0,148,0,1,1,1,144,64,64},
{OP_CONV,13,12,-1,7,1451378622,1,0,0,1.3517016754034743,1,1,144,64,64},
{OP_RELU6,14,13,-1,-1,0,0,157,0,1,1,1,144,64,64},
{OP_CONV,15,14,-1,8,1299006089,1,0,0,1.2097936948941996,1,1,24,64,64},
{OP_ADD,16,10,15,-1,1327889842,0,1874623499,0,0.61834689332303083,0.8729395916569227,1,24,64,64},
{OP_CONV,17,16,-1,9,1073741824,1,0,0,1,1,1,144,64,64},
{OP_RELU6,18,17,-1,-1,0,0,195,0,1,1,1,144,64,64},
{OP_CONV,19,18,-1,10,1112204923,1,0,0,1.0358215520151843,1,1,144,32,32},
{OP_RELU6,20,19,-1,-1,0,0,122,0,1,1,1,144,32,32},
{OP_CONV,21,20,-1,11,1121895083,1,0,0,1.0448462168375652,1,1,32,32,32},
{OP_CONV,22,21,-1,12,1073741824,1,0,0,1,1,1,192,32,32},
{OP_RELU6,23,22,-1,-1,0,0,265,0,1,1,1,192,32,32},
{OP_CONV,24,23,-1,13,1086177134,1,0,0,1.0115812853738708,1,1,192,32,32},
{OP_RELU6,25,24,-1,-1,0,0,191,0,1,1,1,192,32,32},
{OP_CONV,26,25,-1,14,1075014647,1,0,0,1.0011854089508563,1,1,32,32,32},
{OP_ADD,27,21,26,-1,1600491957,0,1303674552,0,0.74528714500478965,0.60707077024586353,1,32,32,32},
{OP_CONV,28,27,-1,15,1073741824,1,0,0,1,1,1,192,32,32},
{OP_RELU6,29,28,-1,-1,0,0,127,0,1,1,1,192,32,32},
{OP_CONV,30,29,-1,16,1250732701,1,0,0,1.1648355992064379,1,1,192,32,32},
{OP_RELU6,31,30,-1,-1,0,0,104,0,1,1,1,192,32,32},
{OP_CONV,32,31,-1,17,1310091605,1,0,0,1.2201178868611655,1,1,32,32,32},
{OP_ADD,33,27,32,-1,1522485264,0,2083372589,-1,0.70896244794677454,0.48507298075982735,1,32,32,32},
{OP_CONV,34,33,-1,18,1073741824,1,0,0,1,1,1,192,32,32},
{OP_RELU6,35,34,-1,-1,0,0,186,0,1,1,1,192,32,32},
{OP_CONV,36,35,-1,19,1073741824,1,0,0,1,1,1,192,16,16},
{OP_RELU6,37,36,-1,-1,0,0,105,0,1,1,1,192,16,16},
{OP_CONV,38,37,-1,20,1293263701,1,0,0,1.2044456799825032,1,1,64,16,16},
{OP_CONV,39,38,-1,21,1073741824,1,0,0,1,1,1,384,16,16},
{OP_RELU6,40,39,-1,-1,0,0,291,0,1,1,1,384,16,16},
{OP_CONV,41,40,-1,22,1088698517,1,0,0,1.0139295062316309,1,1,384,16,16},
{OP_RELU6,42,41,-1,-1,0,0,190,0,1,1,1,384,16,16},
{OP_CONV,43,42,-1,23,1362518147,1,0,0,1.2689439084509486,1,1,64,16,16},
{OP_ADD,44,38,43,-1,2101766178,0,1316815751,0,0.97871114397315939,0.61319011757573416,1,64,16,16},
{OP_CONV,45,44,-1,24,1073741824,1,0,0,1,1,1,384,16,16},
{OP_RELU6,46,45,-1,-1,0,0,228,0,1,1,1,384,16,16},
{OP_CONV,47,46,-1,25,1073741824,1,0,0,1,1,1,384,16,16},
{OP_RELU6,48,47,-1,-1,0,0,102,0,1,1,1,384,16,16},
{OP_CONV,49,48,-1,26,1342118912,1,0,0,1.2499456405639648,1,1,64,16,16},
{OP_ADD,50,44,49,-1,1688639804,0,1296411147,0,0.78633418471400041,0.60368848337434078,1,64,16,16},
{OP_CONV,51,50,-1,27,1073741824,1,0,0,1,1,1,384,16,16},
{OP_RELU6,52,51,-1,-1,0,0,223,0,1,1,1,384,16,16},
{OP_CONV,53,52,-1,28,1073741824,1,0,0,1,1,1,384,16,16},
{OP_RELU6,54,53,-1,-1,0,0,71,0,1,1,1,384,16,16},
{OP_CONV,55,54,-1,29,1909420715,1,0,0,1.7782866160074871,1,1,64,16,16},
{OP_ADD,56,50,55,-1,1502069121,0,1464177020,0,0.69945544039685048,0.68181055582303918,1,64,16,16},
{OP_CONV,57,56,-1,30,1073741824,1,0,0,1,1,1,384,16,16},
{OP_RELU6,58,57,-1,-1,0,0,115,0,1,1,1,384,16,16},
{OP_CONV,59,58,-1,31,1678778905,1,0,0,1.5634846924546191,1,1,384,16,16},
{OP_RELU6,60,59,-1,-1,0,0,112,0,1,1,1,384,16,16},
{OP_CONV,61,60,-1,32,1219584085,1,0,0,1.1358261903127034,1,1,96,16,16},
{OP_CONV,62,61,-1,33,1073741824,1,0,0,1,1,1,576,16,16},
{OP_RELU6,63,62,-1,-1,0,0,247,0,1,1,1,576,16,16},
{OP_CONV,64,63,-1,34,1073741824,1,0,0,1,1,1,576,16,16},
{OP_RELU6,65,64,-1,-1,0,0,96,0,1,1,1,576,16,16},
{OP_CONV,66,65,-1,35,1421691989,1,0,0,1.3240538438161216,1,1,96,16,16},
{OP_ADD,67,61,66,-1,1288978953,0,1722869710,0,0.60022759844335971,0.80227372724057266,1,96,16,16},
{OP_CONV,68,67,-1,36,1073741824,1,0,0,1,1,1,576,16,16},
{OP_RELU6,69,68,-1,-1,0,0,141,0,1,1,1,576,16,16},
{OP_CONV,70,69,-1,37,1537490273,1,0,0,1.4318993993913209,1,1,576,16,16},
{OP_RELU6,71,70,-1,-1,0,0,75,0,1,1,1,576,16,16},
{OP_CONV,72,71,-1,38,1827857067,1,0,0,1.7023245493570964,1,1,96,16,16},
{OP_ADD,73,67,72,-1,1546449698,0,1920341980,0,0.72012175689937075,0.89422891834456741,1,96,16,16},
{OP_CONV,74,73,-1,39,1073741824,1,0,0,1,1,1,576,16,16},
{OP_RELU6,75,74,-1,-1,0,0,116,0,1,1,1,576,16,16},
{OP_CONV,76,75,-1,40,1490107741,1,0,0,1.3877709777863041,1,1,576,8,8},
{OP_RELU6,77,76,-1,-1,0,0,60,0,1,1,1,576,8,8},
{OP_CONV,78,77,-1,41,1138237269,2,0,0,2.120132128397624,1,1,160,8,8},
{OP_CONV,79,78,-1,42,1073741824,1,0,0,1,1,1,960,8,8},
{OP_RELU6,80,79,-1,-1,0,0,232,0,1,1,1,960,8,8},
{OP_CONV,81,80,-1,43,1073741824,1,0,0,1,1,1,960,8,8},
{OP_RELU6,82,81,-1,-1,0,0,95,0,1,1,1,960,8,8},
{OP_CONV,83,82,-1,44,1433278635,1,0,0,1.3348447481791179,1,1,160,8,8},
{OP_ADD,84,78,83,-1,1944857264,0,1867817113,0,0.90564473725131245,0.86977012110315455,1,160,8,8},
{OP_CONV,85,84,-1,45,1073741824,1,0,0,1,1,1,960,8,8},
{OP_RELU6,86,85,-1,-1,0,0,229,0,1,1,1,960,8,8},
{OP_CONV,87,86,-1,46,1349747128,1,0,0,1.2570499705473692,1,1,960,8,8},
{OP_RELU6,88,87,-1,-1,0,0,60,0,1,1,1,960,8,8},
{OP_CONV,89,88,-1,47,1145472341,2,0,0,2.1336085001627607,1,1,160,8,8},
{OP_ADD,90,84,89,-1,1618419767,-2,1096257857,1,0.18840885799340781,1.0209696897052059,1,160,8,8},
{OP_CONV,91,90,-1,48,1073741824,1,0,0,1,1,1,960,8,8},
{OP_RELU6,92,91,-1,-1,0,0,92,0,1,1,1,960,8,8},
{OP_CONV,93,92,-1,49,1479603712,1,0,0,1.377988338470459,1,1,960,8,8},
{OP_RELU6,94,93,-1,-1,0,0,10,0,1,1,1,960,8,8},
{OP_CONV,95,94,-1,50,1681775275,4,0,0,12.530202229817707,1,1,320,8,8},
{OP_CONV,96,95,-1,51,1073741824,1,0,0,1,1,1,1280,8,8},
{OP_RELU6,97,96,-1,-1,0,0,9,0,1,1,1,1280,8,8},
{OP_RESIZE,98,97,-1,-1,0,0,0,0,1,1,1,1280,16,16},
{OP_CONCAT,99,98,73,-1,1905401544,3,1073741824,1,7.098173886724287,1,1,1376,16,16},
{OP_CONV,100,99,-1,52,1073741824,1,0,0,1,1,1,256,16,16},
{OP_RELU,101,100,-1,-1,0,0,0,0,1,1,1,256,16,16},
{OP_CONV,102,101,-1,53,1244217022,1,0,0,1.1587674004395339,1,1,256,16,16},
{OP_RELU,103,102,-1,-1,0,0,0,0,1,1,1,256,16,16},
{OP_RESIZE,104,103,-1,-1,0,0,0,0,1,1,1,256,32,32},
{OP_CONCAT,105,104,33,-1,1481453825,1,1621349444,-1,1.3797113909311853,0.37749983473924165,1,288,32,32},
{OP_CONV,106,105,-1,54,1073741824,1,0,0,1,1,1,128,32,32},
{OP_RELU,107,106,-1,-1,0,0,0,0,1,1,1,128,32,32},
{OP_CONV,108,107,-1,55,1109254741,1,0,0,1.0330739808754779,1,1,128,32,32},
{OP_RELU,109,108,-1,-1,0,0,0,0,1,1,1,128,32,32},
{OP_RESIZE,110,109,-1,-1,0,0,0,0,1,1,1,128,64,64},
{OP_CONCAT,111,110,16,-1,1487466269,1,1104740683,-1,1.3853109154888184,0.25721748426397101,1,152,64,64},
{OP_CONV,112,111,-1,56,1073741824,1,0,0,1,1,1,64,64,64},
{OP_RELU,113,112,-1,-1,0,0,0,0,1,1,1,64,64,64},
{OP_CONV,114,113,-1,57,1316964056,1,0,0,1.2265183555329124,1,1,64,64,64},
{OP_RELU,115,114,-1,-1,0,0,0,0,1,1,1,64,64,64},
{OP_RESIZE,116,115,-1,-1,0,0,0,0,1,1,1,64,128,128},
{OP_CONCAT,117,116,5,-1,1236163149,1,1296220760,-1,1.1512666469947008,0.30179991375224863,1,80,128,128},
{OP_CONV,118,117,-1,58,1073741824,1,0,0,1,1,1,32,128,128},
{OP_RELU,119,118,-1,-1,0,0,0,0,1,1,1,32,128,128},
{OP_CONV,120,119,-1,59,1092395597,1,0,0,1.0173726798220815,1,1,32,128,128},
{OP_RELU,121,120,-1,-1,0,0,0,0,1,1,1,32,128,128},
{OP_RESIZE,122,121,-1,-1,0,0,0,0,1,1,1,32,256,256},
{OP_CONV,123,122,-1,60,1073741824,1,0,0,1,1,1,16,256,256},
{OP_RELU,124,123,-1,-1,0,0,0,0,1,1,1,16,256,256},
{OP_CONV,125,124,-1,61,1073741824,1,0,0,1,1,1,16,256,256},
{OP_RELU,126,125,-1,-1,0,0,0,0,1,1,1,16,256,256},
{OP_CONV,127,126,-1,62,1124565902,1,0,0,1.0473336110376723,1,1,4,256,256},
};
const int NOPS=127, NTENSORS=128, NLAYERS=63, FINAL_TENSOR=127;
/* END INLINE v9_plan.c */

/* BEGIN INLINE v9_runtime.c */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <math.h>
#include <omp.h>
typedef struct {void *p; int type,n,c,h,w;} Tensor;
static void *read_file(const char *root,const char *rel,size_t bytes){char p[4096];snprintf(p,sizeof(p),"%s/%s",root,rel);FILE*f=fopen(p,"rb");if(!f){fprintf(stderr,"open failed %s\n",p);return 0;}void*x=malloc(bytes);if(!x){fclose(f);return 0;}size_t n=fread(x,1,bytes,f);fclose(f);if(n!=bytes){fprintf(stderr,"short read %s\n",p);free(x);return 0;}return x;}
static int32_t rq64(int64_t x,int32_t m,int s){int64_t p=x*(int64_t)m;int64_t q=(llabs(p)+(1LL<<30))>>31;if(p<0)q=-q;if(s>0)q<<=s;else if(s<0){int r=-s;int64_t a=llabs(q);a=(a+(1LL<<(r-1)))>>r;q=q<0?-a:a;}if(q>INT32_MAX)q=INT32_MAX;if(q<INT32_MIN)q=INT32_MIN;return(int32_t)q;}
static int8_t rq8(int64_t x,int32_t m,int s){int32_t q=rq64(x,m,s);return q>127?127:q<-127?-127:(int8_t)q;}
static int8_t fr8(int8_t x,double r){long q=lrint((double)x*r);return q>127?127:q<-127?-127:(int8_t)q;}
static int alloc8(Tensor*t,const Op*o){t->type=8;t->n=o->n;t->c=o->c;t->h=o->h;t->w=o->w;t->p=malloc((size_t)o->n*o->c*o->h*o->w);return t->p!=0;}
static int alloc32(Tensor*t,const Op*o){t->type=32;t->n=o->n;t->c=o->c;t->h=o->h;t->w=o->w;t->p=malloc((size_t)o->n*o->c*o->h*o->w*4);return t->p!=0;}
int v9_run(const char*ip,const char*root,const char*op){Tensor*T=calloc(NTENSORS,sizeof(*T));if(!T)return 2;FILE*f=fopen(ip,"rb");if(!f)return 3;T[0]=(Tensor){malloc(13*256*256),8,1,13,256,256};if(!T[0].p||fread(T[0].p,1,13*256*256,f)!=13*256*256){fclose(f);return 4;}fclose(f);
for(int oi=0;oi<NOPS;oi++){const Op*o=&OPS[oi];Tensor*a=&T[o->a],*z=&T[o->out];if(o->kind==OP_CONV){const Layer*l=&LAYERS[o->layer];size_t wc=(size_t)l->oc*l->icg*l->kh*l->kw;int8_t*w=read_file(root,l->wf,wc);int32_t*b=read_file(root,l->bf,l->oc*4),*mu=read_file(root,l->mf,l->oc*4),*sh=read_file(root,l->sf,l->oc*4);if(!w||!b||!mu||!sh)return 5;int last=o->layer==NLAYERS-1;if(!(last?alloc32(z,o):alloc8(z,o)))return 6;int8_t*x=a->p;int in_c=a->c,in_h=a->h,in_w=a->w;
#pragma omp parallel for collapse(2) schedule(static)
for(int oc=0;oc<l->oc;oc++)for(int oh=0;oh<o->h;oh++)for(int ow=0;ow<o->w;ow++){int g=oc/(l->oc/l->groups);int ic0=g*l->icg;int64_t acc=b[oc];for(int ic=0;ic<l->icg;ic++)for(int kh=0;kh<l->kh;kh++)for(int kw=0;kw<l->kw;kw++){int ih=oh*l->sh-l->ph+kh*l->dh,iw=ow*l->sw-l->pw+kw*l->dw;if((unsigned)ih<(unsigned)in_h&&(unsigned)iw<(unsigned)in_w){int8_t xv=x[((ic0+ic)*in_h+ih)*in_w+iw];xv=fr8(xv,o->r1);int8_t wv=w[(((oc*l->icg+ic)*l->kh+kh)*l->kw+kw)];acc+=(int32_t)xv*(int32_t)wv;}}size_t idx=((size_t)oc*o->h+oh)*o->w+ow;if(last)((int32_t*)z->p)[idx]=rq64(acc,mu[oc],sh[oc]);else((int8_t*)z->p)[idx]=rq8(acc,mu[oc],sh[oc]);}free(w);free(b);free(mu);free(sh);
}else if(o->kind==OP_RELU||o->kind==OP_RELU6){if(!alloc8(z,o))return 7;size_t n=(size_t)o->n*o->c*o->h*o->w;int8_t*x=a->p,*y=z->p;int hi=o->kind==OP_RELU6?o->m2:127;for(size_t i=0;i<n;i++){int v=x[i];y[i]=v<0?0:v>hi?hi:v;}
}else if(o->kind==OP_RESIZE){if(!alloc8(z,o))return 8;int8_t*x=a->p,*y=z->p;for(int c=0;c<o->c;c++)for(int h=0;h<o->h;h++)for(int w0=0;w0<o->w;w0++){int ih=h*a->h/o->h,iw=w0*a->w/o->w;y[(c*o->h+h)*o->w+w0]=x[(c*a->h+ih)*a->w+iw];}
}else if(o->kind==OP_ADD){Tensor*bb=&T[o->b];if(!alloc8(z,o))return 9;size_t n=(size_t)o->n*o->c*o->h*o->w;int8_t*x=a->p,*y=bb->p,*q=z->p;for(size_t i=0;i<n;i++){int v=fr8(x[i],o->r1)+fr8(y[i],o->r2);q[i]=v>127?127:v<-127?-127:v;}
}else if(o->kind==OP_CONCAT){Tensor*bb=&T[o->b];if(!alloc8(z,o))return 10;int8_t*q=z->p,*x=a->p,*y=bb->p;size_t plane=(size_t)o->h*o->w;for(int c=0;c<a->c;c++)for(size_t i=0;i<plane;i++)q[(size_t)c*plane+i]=fr8(x[(size_t)c*plane+i],o->r1);for(int c=0;c<bb->c;c++)for(size_t i=0;i<plane;i++)q[(size_t)(a->c+c)*plane+i]=fr8(y[(size_t)c*plane+i],o->r2);}}
Tensor*z=&T[FINAL_TENSOR];if(z->type!=32||z->c!=4)return 11;uint8_t*mask=malloc((size_t)z->h*z->w);int32_t*v=z->p;for(int p=0;p<z->h*z->w;p++){int best=0;for(int c=1;c<4;c++)if(v[c*z->h*z->w+p]>v[best*z->h*z->w+p])best=c;mask[p]=best;}f=fopen(op,"wb");if(!f)return 12;fwrite(mask,1,(size_t)z->h*z->w,f);fclose(f);free(mask);for(int i=0;i<NTENSORS;i++)free(T[i].p);free(T);return 0;}
/* END INLINE v9_runtime.c */

/* BEGIN INLINE main.c */
#include <stdio.h>
int main(int argc,char**argv){if(argc!=4){fprintf(stderr,"Usage: %s INPUT_INT8.bin PACKAGE_ROOT OUTPUT_U8.bin\n",argv[0]);return 2;}return v9_run(argv[1],argv[2],argv[3]);}
/* END INLINE main.c */

