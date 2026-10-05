# Data inventory (measured in this checkout)

## Competition rasters (via git mirrors, sha256-verified 2026-10-05)
| file | bytes | sha256 | facts |
|---|---|---|---|
| training_features.tif | 418,912,844 | `4371c82e…23bc5` | 19× float32, nodata −3.4e38 |
| labels.tif / existing_faults.tif | 425,830 | `7ba308cc…25ae4093` | int8 (−1/0/1), 60,988 positives; byte-identical pair |
| sample_submission.tif | 1,599,597 | `2176d08e…d35cbc` | float32, NaN outside; 5,167,373 finite; 60,988 ones |

## Band map (raster descriptions, read here)
1 mag_anom · 2 rtp · 3 tmi_hg · 4 geod_2ndinv · 5 iso_grav_anom_slope ·
6 tc · 7 geod_shearrate · 8 geod_dilaterate · 9 tmi_vg · 10 deq_n100a15 ·
11 iso_grav_anom_vg · 12 det_elev · 13 iso_grav_anom · 14 tmi ·
15 depth_to_base_surf · 16 ieq_n100a15 · 17 cond_surf · 18 iso_grav_anom_hg ·
19 det_elev_slope

## External (owner mirrors of official releases)
- derived_sgmc_faults_100m_u8.tif (USGS SGMC faults → PRIMARY instrument)
- gdr_wellspring_in_footprint.csv (GDR 1391, CC-BY-4.0)
- 2m_temperature_probe zip + paleo_geothermal zip (H42-4/H42-5 queued use)
- geodawn rad/extensions + lidar scarp u8 (available, unused by H42-1)

## Grid
EPSG:32611 · 100 m · 3730×3292 (12,279,160) ·
Affine(100, 0, 243350, 0, −100, 4508550).
