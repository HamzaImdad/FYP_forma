#!/bin/bash
# ExerVision — Batch video downloader with live progress
# Usage: bash scripts/download_videos.sh
# Pause: Ctrl+C (safe — resumes where it left off on next run)
# Resume: just run the same command again

PROJECT_ROOT="/c/Users/Hamza/OneDrive - University of Greenwich/Documents/fyp github"
DATA_DIR="$PROJECT_ROOT/data/datasets/youtube"
LOG_FILE="$PROJECT_ROOT/data/datasets/youtube/download_log.txt"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# Track totals
TOTAL_URLS=0
TOTAL_DONE=0
TOTAL_FAIL=0
TOTAL_SKIP=0

# Initialize log
echo "=== Download started: $(date) ===" >> "$LOG_FILE"

declare -A EXERCISES
EXERCISES=(
  ["squat"]="LmFitmO_ANw CVrTmUF9yJ8 7FF6VLej_bY dO-51htz_eA LSj280OEKUI 8Kls95w2jFA irA7MTz96ho I7eR8d0cNQQ W73Mc0Gil9A 9KzZD_n2r64 dBJry3tcX0Q my0tLDaWyDU m_1Id1VUwgQ gcNh17Ckjgg niWFDg6wyD0 GQ5jj_zH2uA 3PRwtVpyslo QifjltKUMCk VWm_SY6pO_Y HFnSsLIB7a4 nSgSbTM4JS4 X8DnpBl7Hwk U11z1PmtohU QKKXd6celO0 q1fCgfieNEs 47hOJe2_AqI iIIqHDGH0B8 WT4RVML0Ytw XHuYqVV9qmY H-kPX7cirNU -sM5af2545M Mu7aVOjEBdA 9W5KAqHfDe8 quzkqPHF7MY ZuVC1V30IEw byxWus7BwfQ 3aNpOOZR15I MeIiIdhvXT4 H1H1WKSm5CU QilYZ20kVfM JANvVVsyZJE 4KmY44Xsg2w H5VYU6t_w9o YaXPRqUwItQ nmUof3vszxM"
  ["pushup"]="A7sQ3IzDKvU Ac6WhCsMArE eiMOxvZKyvM Jf5_PJCFs-g 9-DlYB4vO4U Pp44Y390Ffs -f7HvbPcSg4 rruHM_sB2Hc m1W1BGpPGMo MO10KOoQx5E 1Bj5PPxgEwo FeXQjPhqdiI zkU6Ok44_CI ZeuxTYGv0g4 BBSkdTaEEQs xxOdD929ty8 b0xFZ2HtENM htKUAKYMC-I gLsarcKlS6c rzBcOh6vVHc JUAV7bryQM0 hwnsLAFDuHA I9fsqKE5XHo Z88Rl5bpnmI kM47_kIXGAY 5aRHy5ZPjwk ryncZFQCB8I i9sTjhN4Z3M e22iFFXY8vw _VhqLJgBecs Yd1grZkAark bOHpI9LEea8 ck1X4o0B5nM TDmg3tFGOvE qqyquVlh03Y cOiT2mELfCY monNCi-43Qg XOg81X0Tckw bt5b9x9N0KU P3D2PPMYWjk"
  ["pullup"]="Hdc7Mw6BIEE D-TL5hDQucI _hd2oTEOA1U MhokcbRLP5w iBtL9nX2qOs 5WHdim80e7o syS4M1G-rII hkBcTOa21oQ sIvJTfGxdFo Y3ntNsIS2Q8 x3NPAxiMRPw xLyFaI491ho bb8_5vZV5dU p40iUjf02j0 rmdn5X_KLkY ScGObDq1DSo GOVt1rXy1QQ SUfu0HCjZzI hX25-nMbjvs s6acFNvGu5s fO3dKSQayfg 6zyx46Vpato FWKi87ss8vs bHC16skSN6Q vajKCFo0c6E xewBaDLnfBQ 6GWT7GLXE3c jFmCrA6fo78 XeErfmGSwfE EihD_pt2AFA RvT3beYzPrI GdR0jI6mzME tTF2Cu05tNU rVRiy_HVz2g e1YSApl-QcM"
  ["overhead_press"]="nNMR9fRGRjQ KP1sYz2VICk 8dacy5hjaE8 vlFGTI5JzjI F3QY5vMz_6I GGpArq725ls 2N5P_iWkluQ iJ0py9JQIZY wJlKUo2-P4o 0n86YPrgDBs CnBmiBqp-AI Wqq43dKW1TU 3R14MnZbcpw Gu1t7X2yq4M rO_iEImwHyo qEwKCR5JCog nHboL27_Sn0 BAZkFGeUy5U _hDKR72wckw wol7Hko8RhY yEbczj_FGyI xN08c9hSBw0 GFblCmuEE18 OOe_HrNnQWw ris9tKqMwgU bN1YwwqPnnM JAA855alyw4"
  ["deadlift"]="XxWcirHIwVo MBbyAqvTNkU m__vJ_Fqsy4 ntr64W6ZWB0 e2cbNBLH5SQ p2OPUi4xGrM TM1mpvglJq4 BSpHBgvOGps mlOUL-Pkzls g2Xl1zJeArs GxsLrTzyGUU 19ZeTrLZdyQ WFUOtnI1jwk 87UE4nT-niE UyRiw2b8yI4 NE91CqP1JKE dBXnS-c5r0Q vRKDvt695pg _Jbl6ZpVGzI F4mHL08LXIg 5zmlnbWb-g4 _oyxCn2iSjU 5bJEigM5iVg FUwsp0OVyVM uhghy9pFIPY ZEnWV4kguKc XsrD5y8EIKU LGIS9vs65Sk wT_fCx2hZVw"
  ["lunge"]="ugW5I-a5A-Q mBhqeQm8RdM py8vxF9b46Y g8x9ck225ZE Z6R8A5tcrTc jTodqKdZnm0 qQyCK_rxzN0 fydLSJlGx-0 hsO1dZjcPj8 -yxNPwWlr8I SXYrUTUwFoc xvC10-eCuXs 3XDriUn0udo 1LuRcKJMn8w krXwDPKKiSM cHWPbW25py4 G_cv0yXpNPU C_P3Q-PssvY DKAILCp9POg D4cM8BdpOiY Es49QJkx3JE BYWwfljY57w"
  ["plank"]="6LqqeBtFn9M A2b2EmIg0dA BQu26ABuVS0 kL_NJAkCQBg EtO0CutfR8Y F-nQ_KJgfCY 7NS4vNWA78o ybQLwX0KWls xxY_7iffTi8 1RlDDa0N9Ws 8KavLXXs24U gvHVdNVBu6s A9bI8BzqjMg mH5Sfb_KTGg X9CbjvtVuOk 4hQmAsF9iL4 hS4YLDWNKUw fu540x6vISA Iv75ErM4wCI jYydu_1FOHs qdy1nPYw1Eo TvxNkmjdhMM"
  ["bicep_curl"]="i1YgFZB6alI QZEqB6wUPxQ ymCcs3lhiLs DCe8f6vMe9A wzvOiPizqSc yTWO2th-RIY LPMaGWJgjgc 8XLxfXROrTo sYV-ki-1blM XE_pHwbst04 5FAuyZuvJFg bAWLx7PPK10 TwD-YGVP4Bk av7-8igSXTs Nkl8WnH6tDU BRVDS6HVR9Q OPqe0kCxmR8"
  ["tricep_dip"]="KoS_NMmuxMM Tw0axi-Jlqc vi1-BOcj3cQ l41SoWZiowI AWz_7B1cch0 jdFzYGmvDyg j_WpuVY3wbo e5Gyc1D_BxM EBnq0A5L_wo TrJVszDm7ik iYbK_uiwCUg cevUzvuMNmY S787NKGhD88 KbPBe1osXP4"
)

# Order by priority (biggest gap first)
EXERCISE_ORDER="squat pushup pullup overhead_press deadlift lunge plank bicep_curl tricep_dip"

# Count total
for ex in $EXERCISE_ORDER; do
  ids=(${EXERCISES[$ex]})
  TOTAL_URLS=$((TOTAL_URLS + ${#ids[@]}))
done

echo ""
echo -e "${BOLD}========================================${NC}"
echo -e "${BOLD}  ExerVision Video Downloader${NC}"
echo -e "${BOLD}  Total: $TOTAL_URLS videos across 9 exercises${NC}"
echo -e "${BOLD}  Press Ctrl+C to pause (safe to resume)${NC}"
echo -e "${BOLD}========================================${NC}"
echo ""

GLOBAL_COUNT=0

for ex in $EXERCISE_ORDER; do
  ids=(${EXERCISES[$ex]})
  EX_TOTAL=${#ids[@]}
  EX_DONE=0
  EX_FAIL=0
  EX_SKIP=0

  OUT_DIR="$DATA_DIR/$ex/correct"
  mkdir -p "$OUT_DIR"

  echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
  echo -e "${CYAN}  $ex ($EX_TOTAL videos)${NC}"
  echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

  for vid_id in "${ids[@]}"; do
    GLOBAL_COUNT=$((GLOBAL_COUNT + 1))

    # Check if already downloaded (any extension)
    existing=$(ls "$OUT_DIR"/${vid_id}.* 2>/dev/null | head -1)
    if [ -n "$existing" ]; then
      EX_SKIP=$((EX_SKIP + 1))
      TOTAL_SKIP=$((TOTAL_SKIP + 1))
      echo -e "  ${YELLOW}[SKIP]${NC} [$GLOBAL_COUNT/$TOTAL_URLS] $ex/$vid_id (already exists)"
      continue
    fi

    echo -e "  ${GREEN}[DOWN]${NC} [$GLOBAL_COUNT/$TOTAL_URLS] $ex/$vid_id ..."

    "C:/Users/Hamza/AppData/Local/Programs/Python/Python310/Scripts/yt-dlp.exe" \
      -f "bestvideo[height<=720]+bestaudio/best[height<=720]" \
      --merge-output-format mp4 \
      -o "$OUT_DIR/%(id)s.%(ext)s" \
      --no-warnings \
      --progress \
      "https://www.youtube.com/watch?v=$vid_id" 2>&1

    if [ $? -eq 0 ]; then
      EX_DONE=$((EX_DONE + 1))
      TOTAL_DONE=$((TOTAL_DONE + 1))
      echo "$ex $vid_id OK" >> "$LOG_FILE"
    else
      EX_FAIL=$((EX_FAIL + 1))
      TOTAL_FAIL=$((TOTAL_FAIL + 1))
      echo -e "  ${RED}[FAIL]${NC} $vid_id"
      echo "$ex $vid_id FAIL" >> "$LOG_FILE"
    fi
  done

  echo -e "  ${BOLD}$ex summary: ${GREEN}$EX_DONE downloaded${NC}, ${YELLOW}$EX_SKIP skipped${NC}, ${RED}$EX_FAIL failed${NC}"
  echo ""
done

echo -e "${BOLD}========================================${NC}"
echo -e "${BOLD}  COMPLETE${NC}"
echo -e "  ${GREEN}Downloaded: $TOTAL_DONE${NC}"
echo -e "  ${YELLOW}Skipped (already had): $TOTAL_SKIP${NC}"
echo -e "  ${RED}Failed: $TOTAL_FAIL${NC}"
echo -e "  Total: $TOTAL_URLS"
echo -e "${BOLD}========================================${NC}"
echo "=== Download finished: $(date) ===" >> "$LOG_FILE"
