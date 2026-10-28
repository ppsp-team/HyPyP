#!/usr/bin/env bash
# Open the review's issues on ppsp-team/HyPyP, each fix branch linked to its issue.
#
# For every fix branch: push its commits under a temporary ref, open the issue
# from issues/branch-NN-*.md, create the real branch at the tip through
# GitHub's createLinkedBranch (which is what links a branch to an issue in the
# "Development" panel), then drop the temporary ref. Then the six deferred
# issues, then the umbrella issue with every number filled in. Also pushes the
# review bundle branch if it exists locally.
#
# Usage: bash create_issues.sh [clone dir]      DRY_RUN=1 prints instead of doing.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
CLONE=${1:-$HOME/Github/HyPyP}
OWNER=ppsp-team
NAME=HyPyP
DRY=${DRY_RUN:-0}

run() { if [ "$DRY" = 1 ]; then echo "+ $*"; else "$@"; fi; }

cd "$CLONE"
REPO_ID=$(gh api graphql -f query="{repository(owner:\"$OWNER\",name:\"$NAME\"){id}}" --jq .data.repository.id)

issue_title() { head -1 "$1" | sed -E 's/^# //; s/^Issue draft[^:]*: //'; }
issue_body() { tail -n +2 "$1"; }

create_issue() {  # file -> prints the issue number
    local file=$1 url
    if [ "$DRY" = 1 ]; then echo "+ gh issue create --title '$(issue_title "$file")' --body-file ..." >&2; echo 0; return; fi
    url=$(issue_body "$file" | gh issue create --repo "$OWNER/$NAME" --title "$(issue_title "$file")" --body-file -)
    echo "${url##*/}"
}

link_branch() {  # issue number, branch
    local number=$1 branch=$2 issue_id oid tmp
    oid=$(git rev-parse "$branch")
    tmp="refs/heads/tmp/${branch//\//-}"
    run git push -q origin "$branch:$tmp"
    if [ "$DRY" = 1 ]; then echo "+ createLinkedBranch(issue #$number, $branch @ ${oid:0:7})"; return; fi
    issue_id=$(gh api graphql -f query="{repository(owner:\"$OWNER\",name:\"$NAME\"){issue(number:$number){id}}}" --jq .data.repository.issue.id)
    gh api graphql -f query="mutation{createLinkedBranch(input:{issueId:\"$issue_id\",oid:\"$oid\",name:\"$branch\",repositoryId:\"$REPO_ID\"}){linkedBranch{ref{name}}}}" --jq .data.createLinkedBranch.linkedBranch.ref.name
    git push -q origin ":$tmp"
    git fetch -q origin "$branch"
    git branch -q -u "origin/$branch" "$branch"
}

BRANCHES=(
    fix/stats-cluster-maps fix/xdf-markers-scale fix/analyses-nmplv
    fix/mvarica-arity-merge fix/prep-ar-local-alignment fix/utils-virtual-epoch
    fix/stats-adjacency-layout fix/stats-paired-test fix/stats-statscond
    test/reference-suite docs/conventions
)
declare -a NUMBERS=()
for i in "${!BRANCHES[@]}"; do
    n=$(printf "%02d" $((i + 1)))
    file=$(ls "$HERE"/branch-$n-*.md)
    number=$(create_issue "$file")
    NUMBERS+=("$number")
    echo "branch ${BRANCHES[$i]} -> issue #$number"
    link_branch "$number" "${BRANCHES[$i]}"
done

declare -a DEFERRED=()
for file in F13-F14-F15-fnirs-wavelet F17-F18-xdf-time-axis F10-xwt-smoothing F20-viz ci-packaging section5-method-decisions; do
    number=$(create_issue "$HERE/$file.md")
    DEFERRED+=("$number")
    echo "deferred $file -> issue #$number"
done

# the umbrella, with every number filled in
umbrella=$(mktemp)
cp "$HERE/00-review-and-fix-plan.md" "$umbrella"
for i in "${!BRANCHES[@]}"; do
    sed -i -E "s|^- \[ \] $((i + 1))\. |- [ ] $((i + 1)). #${NUMBERS[$i]} |" "$umbrella"
done
labels=("F13, F14, F15 and the fNIRS MAJOR items" "F17, F18" "F10 and" "F20 and" "CI with" "the nine method")
for i in "${!labels[@]}"; do
    sed -i -E "s|^- \[ \] (${labels[$i]})|- [ ] #${DEFERRED[$i]} \1|" "$umbrella"
done
sed -i 's|^(The issue numbers go here once they exist.)$||' "$umbrella"
number=$(create_issue "$umbrella")
echo "umbrella -> issue #$number"

if git show-ref -q --verify refs/heads/review/2026-10-13; then
    run git push -u origin review/2026-10-13
fi
echo "done; attach REVIEW-2026-10-13.md to issue #$number from the web UI, or point to the review bundle branch"
