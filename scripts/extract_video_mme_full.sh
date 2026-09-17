#!/bin/bash
# 解压全量 Video-MME；要求先准备独立校验清单，不假定下载已完成校验：
#   - 20 个 videos_chunked_*.zip 均为独立 zip（内部路径 data/<video_id>.mp4），
#     用 unzip -j 去掉内部路径前缀，平铺解压到 data/videos/video_mme/（与 selected50 合并）
#   - subtitle.zip 内部路径 subtitle/<video_id>.srt，解压到 data/subtitles/video_mme/
#   - 标注 parquet 保留在 data/archives/video_mme_full/videomme/
# 幂等：unzip -n 不覆盖已存在文件，重复执行安全。
# 用法：bash scripts/extract_video_mme_full.sh
set -euo pipefail

SRC=/home/admin/projects/explore_video_qa/data/archives/video_mme_full
VIDEOS=/home/admin/projects/explore_video_qa/data/videos/video_mme
SUBS=/home/admin/projects/explore_video_qa/data/subtitles/video_mme

# 1. 校验文件大小（清单：data/manifests/video_mme_full_filelist.txt）
MANIFEST=/home/admin/projects/explore_video_qa/data/manifests/video_mme_full_filelist.txt
[[ -s "$MANIFEST" ]] || { echo "Missing or empty manifest: $MANIFEST; extraction stopped" >&2; exit 1; }
echo "=== verifying file sizes against manifest ==="
while IFS=$'\t' read -r path expected; do
    [ -z "$path" ] && continue
    actual=$(stat -c%s "$SRC/$path" 2>/dev/null || echo 0)
    if [ "$actual" != "$expected" ]; then
        echo "SIZE MISMATCH: $path have $actual expected $expected" >&2
        exit 1
    fi
done < "$MANIFEST"
echo "all $(grep -c . /home/admin/projects/explore_video_qa/data/manifests/video_mme_full_filelist.txt) files verified"

# 2. 解压视频（-j 平铺，-n 不覆盖已有文件）
shopt -s nullglob
chunks=("$SRC"/videos_chunked_*.zip)
(( ${#chunks[@]} == 20 )) || { echo "Expected 20 video archives, found ${#chunks[@]}" >&2; exit 1; }
mkdir -p "$VIDEOS"
for chunk in "${chunks[@]}"; do
    echo "[$(date '+%F %T')] extracting $(basename "$chunk") ..."
    if ! unzip -n -q -j "$chunk" -d "$VIDEOS"; then
        echo "UNZIP FAILED: $chunk" >&2
        exit 1
    fi
done
echo "mp4 count in $VIDEOS: $(ls "$VIDEOS"/*.mp4 | wc -l)"

# 3. 解压字幕
mkdir -p "$SUBS"
unzip -n -q -j "$SRC/subtitle.zip" -d "$SUBS"
echo "srt count in $SUBS: $(ls "$SUBS"/*.srt | wc -l)"

echo "=== EXTRACT DONE $(date '+%F %T') ==="
