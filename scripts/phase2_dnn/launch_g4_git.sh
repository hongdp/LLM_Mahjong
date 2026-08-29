#!/bin/bash
# Create a G4 flex VM, git-clone the public repo at a pinned commit, launch ONE run.
# Usage: launch_g4_git.sh <vm> <zone> <run_name> <git_sha> -- <trainer args...>
set -uo pipefail
VM="$1"; ZONE="$2"; N="$3"; SHA="$4"; shift 4; [ "${1:-}" = "--" ] && shift
# the trainer defaults exp_dir to a timestamped dir nobody syncs — pin it to the run name
case " $* " in *" --exp_dir "*) ;; *) set -- "$@" --exp_dir "experiments/$N";; esac
P=workstation-185016
# hard gate: the pinned commit must already be on GitHub (the VM clones it)
git fetch -q origin
git branch -r --contains "$SHA" 2>/dev/null | grep -q "origin/" || { echo "ABORT: $SHA is not on any origin branch — push first"; exit 2; }
if ! gcloud compute instances describe $VM --zone=$ZONE --project=$P >/dev/null 2>&1; then
  for a in 1 2 3; do
    gcloud compute instances create $VM --project=$P --zone=$ZONE --machine-type=g4-standard-48 \
      --image-family=common-cu129-ubuntu-2204-nvidia-580 --image-project=deeplearning-platform-release \
      --boot-disk-size=50GB --boot-disk-type=hyperdisk-balanced --maintenance-policy=TERMINATE \
      --provisioning-model=FLEX_START --instance-termination-action=DELETE --max-run-duration=12h \
      --scopes=storage-rw,logging-write --metadata="install-nvidia-driver=True" >/dev/null 2>&1 && break
    echo "create attempt $a failed"; sleep 30
  done
fi
for t in $(seq 1 30); do gcloud compute ssh $VM --zone=$ZONE --command='echo SSH_OK' -- -o ConnectTimeout=10 2>/dev/null | grep -q SSH_OK && break; sleep 15; done
TMP=$(mktemp)
cat > "$TMP" <<EOS
#!/bin/bash
set -e
if [ ! -d \$HOME/LLM_Mahjong/.git ]; then git clone -q https://github.com/hongdp/LLM_Mahjong.git \$HOME/LLM_Mahjong; fi
cd \$HOME/LLM_Mahjong && git fetch -q origin && git checkout -q $SHA && echo "code at \$(git rev-parse --short HEAD)"
${CKPT_GS:+gsutil -q cp $CKPT_GS /tmp/resume.pt && echo "CKPT_OK \$(ls -la /tmp/resume.pt | awk '{print \$5}')"}
${DATA_GS:+sudo apt-get install -y -q zstd >/dev/null 2>&1 || true; gsutil -q cp $DATA_GS /tmp/data.tar.zst && mkdir -p \$HOME/LLM_Mahjong/data/tenhou && tar --zstd -xf /tmp/data.tar.zst -C \$HOME/LLM_Mahjong/data/tenhou && echo "DATA_OK \$(find \$HOME/LLM_Mahjong/data/tenhou/raw -name '*.mjlog' | wc -l) logs"}
nohup bash \$HOME/LLM_Mahjong/scripts/phase2_dnn/run_dnn_cloud.sh $N python $* > \$HOME/${N}_nohup.log 2>&1 &
disown; echo "LAUNCHED $N pid=\$!"
EOS
for a in 1 2 3; do gcloud compute scp "$TMP" $VM:~/launch_run.sh --zone=$ZONE -q 2>/dev/null && break; sleep 15; done
rm -f "$TMP"
gcloud compute ssh $VM --zone=$ZONE --command='bash ~/launch_run.sh' -- -o ConnectTimeout=15 2>/dev/null | tail -2
