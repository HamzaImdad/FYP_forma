#!/bin/bash
# Download training videos for all weak exercises
# Uses yt-dlp search to find short clips with clear form

PROJECT_DIR="/c/Users/Hamza/OneDrive - University of Greenwich/Documents/fyp github"
INCORRECT_DIR="$PROJECT_DIR/data/datasets/youtube_incorrect"
CORRECT_DIR="$PROJECT_DIR/data/datasets/youtube"

MAX_DUR=90
PER_QUERY=3

download_videos() {
    local exercise="$1"
    local label="$2"
    local query="$3"
    local outdir="$4"

    mkdir -p "$outdir"
    echo "  >> [$exercise/$label] '$query'"

    yt-dlp "ytsearch${PER_QUERY}:${query}" \
        --match-filter "duration<=${MAX_DUR}" \
        --max-downloads $PER_QUERY \
        -f "bestvideo[height<=720]+bestaudio/best[height<=720]" \
        --merge-output-format mp4 \
        -o "${outdir}/%(id)s.%(ext)s" \
        --no-playlist \
        --no-overwrites \
        --quiet --no-warnings \
        2>/dev/null
}

echo "============================================"
echo "  DOWNLOADING INCORRECT FORM VIDEOS"
echo "============================================"

# SQUAT - incorrect
for q in \
    "squat bad form gym" \
    "squat form fail compilation short" \
    "squat knee cave gym" \
    "squat butt wink fail" \
    "squat rounded back gym fail" \
    "bad squat technique gym" \
    "worst squat form" \
    "squat form check bad" \
    "squat ego lifting fail" \
    "squat depth fail gym" \
    "beginner squat mistakes gym" \
    "squat half rep gym"; do
    download_videos "squat" "incorrect" "$q" "$INCORRECT_DIR/squat"
done
echo "  Squat incorrect: $(ls "$INCORRECT_DIR/squat/"*.mp4 2>/dev/null | wc -l) total"

# DEADLIFT - incorrect
for q in \
    "deadlift bad form gym" \
    "deadlift rounded back fail" \
    "deadlift form fail compilation short" \
    "deadlift hitching gym" \
    "deadlift lockout fail" \
    "worst deadlift form ever" \
    "deadlift ego lift fail" \
    "deadlift lower back round gym" \
    "deadlift form check bad" \
    "beginner deadlift mistakes" \
    "deadlift cat back gym" \
    "deadlift jerking fail"; do
    download_videos "deadlift" "incorrect" "$q" "$INCORRECT_DIR/deadlift"
done
echo "  Deadlift incorrect: $(ls "$INCORRECT_DIR/deadlift/"*.mp4 2>/dev/null | wc -l) total"

# BENCH PRESS - incorrect
for q in \
    "bench press bad form gym" \
    "bench press fail compilation short" \
    "bench press elbow flare gym" \
    "bench press bouncing reps" \
    "bench press no leg drive" \
    "worst bench press form" \
    "bench press ego lift fail" \
    "bench press form check bad" \
    "bench press shoulder pain form" \
    "beginner bench press mistakes" \
    "bench press butt off bench" \
    "bench press half rep gym"; do
    download_videos "bench_press" "incorrect" "$q" "$INCORRECT_DIR/bench_press"
done
echo "  Bench press incorrect: $(ls "$INCORRECT_DIR/bench_press/"*.mp4 2>/dev/null | wc -l) total"

# PUSHUP - incorrect
for q in \
    "pushup bad form gym" \
    "pushup form fail" \
    "pushup sagging hips" \
    "pushup half reps" \
    "pushup wrong technique" \
    "worst pushup form" \
    "pushup form check bad" \
    "beginner pushup mistakes" \
    "pushup elbows too wide fail" \
    "pushup head forward mistake" \
    "pushup no full range motion" \
    "push up incorrect demonstration"; do
    download_videos "pushup" "incorrect" "$q" "$INCORRECT_DIR/pushup"
done
echo "  Pushup incorrect: $(ls "$INCORRECT_DIR/pushup/"*.mp4 2>/dev/null | wc -l) total"

# OVERHEAD PRESS - incorrect
for q in \
    "overhead press bad form" \
    "shoulder press fail gym" \
    "overhead press lean back too much" \
    "military press bad form" \
    "overhead press elbow flare" \
    "worst overhead press form" \
    "OHP ego lift fail" \
    "overhead press form check bad" \
    "shoulder press mistakes gym" \
    "beginner overhead press mistakes" \
    "overhead press lower back arch" \
    "standing press bad technique"; do
    download_videos "overhead_press" "incorrect" "$q" "$INCORRECT_DIR/overhead_press"
done
echo "  Overhead press incorrect: $(ls "$INCORRECT_DIR/overhead_press/"*.mp4 2>/dev/null | wc -l) total"

# PLANK - incorrect
for q in \
    "plank bad form" \
    "plank form mistakes" \
    "plank hips sagging" \
    "plank butt too high" \
    "plank wrong technique demonstration" \
    "worst plank form" \
    "plank form check bad" \
    "beginner plank mistakes" \
    "plank neck position mistake" \
    "plank shoulder mistake" \
    "plank incorrect body position" \
    "plank common form errors"; do
    download_videos "plank" "incorrect" "$q" "$INCORRECT_DIR/plank"
done
echo "  Plank incorrect: $(ls "$INCORRECT_DIR/plank/"*.mp4 2>/dev/null | wc -l) total"

# TRICEP DIP - incorrect
for q in \
    "tricep dip bad form" \
    "dip form fail gym" \
    "dip shoulder impingement form" \
    "tricep dip too deep mistake" \
    "dip elbow flare fail" \
    "worst dip form gym" \
    "bench dip bad form" \
    "parallel bar dip mistakes" \
    "beginner dip mistakes gym" \
    "dip forward lean too much" \
    "dip form check bad" \
    "tricep dip wrong technique"; do
    download_videos "tricep_dip" "incorrect" "$q" "$INCORRECT_DIR/tricep_dip"
done
echo "  Tricep dip incorrect: $(ls "$INCORRECT_DIR/tricep_dip/"*.mp4 2>/dev/null | wc -l) total"

echo ""
echo "============================================"
echo "  DOWNLOADING CORRECT FORM VIDEOS"
echo "============================================"

# SQUAT - correct
for q in \
    "perfect squat form side view" \
    "squat proper technique gym" \
    "squat good form demonstration" \
    "barbell squat correct form" \
    "squat textbook form" \
    "squat form check good"; do
    download_videos "squat" "correct" "$q" "$CORRECT_DIR/squat/correct"
done
echo "  Squat correct: $(ls "$CORRECT_DIR/squat/correct/"*.mp4 2>/dev/null | wc -l) total"

# DEADLIFT - correct
for q in \
    "perfect deadlift form side view" \
    "deadlift proper technique gym" \
    "deadlift good form demonstration" \
    "conventional deadlift correct form" \
    "deadlift textbook form" \
    "deadlift form check good"; do
    download_videos "deadlift" "correct" "$q" "$CORRECT_DIR/deadlift/correct"
done
echo "  Deadlift correct: $(ls "$CORRECT_DIR/deadlift/correct/"*.mp4 2>/dev/null | wc -l) total"

# BENCH PRESS - correct
for q in \
    "perfect bench press form" \
    "bench press proper technique gym" \
    "bench press good form demonstration" \
    "bench press correct form side view" \
    "bench press textbook form" \
    "bench press form check good"; do
    download_videos "bench_press" "correct" "$q" "$CORRECT_DIR/bench_press/correct"
done
echo "  Bench press correct: $(ls "$CORRECT_DIR/bench_press/correct/"*.mp4 2>/dev/null | wc -l) total"

# PUSHUP - correct
for q in \
    "perfect pushup form" \
    "pushup proper technique" \
    "pushup good form demonstration" \
    "military pushup correct form" \
    "pushup textbook form" \
    "strict pushup form"; do
    download_videos "pushup" "correct" "$q" "$CORRECT_DIR/pushup/correct"
done
echo "  Pushup correct: $(ls "$CORRECT_DIR/pushup/correct/"*.mp4 2>/dev/null | wc -l) total"

# OVERHEAD PRESS - correct
for q in \
    "perfect overhead press form" \
    "overhead press proper technique" \
    "military press good form" \
    "shoulder press correct form" \
    "standing press textbook form" \
    "OHP correct technique"; do
    download_videos "overhead_press" "correct" "$q" "$CORRECT_DIR/overhead_press/correct"
done
echo "  Overhead press correct: $(ls "$CORRECT_DIR/overhead_press/correct/"*.mp4 2>/dev/null | wc -l) total"

# PLANK - correct
for q in \
    "perfect plank form" \
    "plank proper technique" \
    "plank correct form side view" \
    "plank good form 30 seconds" \
    "plank textbook form" \
    "strict plank form"; do
    download_videos "plank" "correct" "$q" "$CORRECT_DIR/plank/correct"
done
echo "  Plank correct: $(ls "$CORRECT_DIR/plank/correct/"*.mp4 2>/dev/null | wc -l) total"

# TRICEP DIP - correct
for q in \
    "perfect tricep dip form" \
    "dip proper technique" \
    "parallel bar dip correct form" \
    "bench dip good form" \
    "tricep dip correct technique" \
    "dip form demonstration good"; do
    download_videos "tricep_dip" "correct" "$q" "$CORRECT_DIR/tricep_dip/correct"
done
echo "  Tricep dip correct: $(ls "$CORRECT_DIR/tricep_dip/correct/"*.mp4 2>/dev/null | wc -l) total"

echo ""
echo "============================================"
echo "  DOWNLOAD COMPLETE - FINAL COUNTS"
echo "============================================"
for ex in squat deadlift bench_press pushup overhead_press plank tricep_dip; do
    inc=$(ls "$INCORRECT_DIR/$ex/"*.mp4 2>/dev/null | wc -l)
    cor=$(ls "$CORRECT_DIR/$ex/correct/"*.mp4 2>/dev/null | wc -l)
    echo "  $ex: correct=$cor, incorrect=$inc"
done
