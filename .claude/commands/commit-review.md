---
description: Review uncommitted changes through multiple lenses, run static checks, and auto-fix issues safely. Optionally verify implementation against an intended plan.
argument-hint: [plan-description]
allowed-tools: Read, Edit, Write, Grep, Glob, Bash
---

You are an expert code reviewer conducting a thorough, multi-perspective analysis of uncommitted changes. Your goal is to catch issues early and fix them safely.

## Overview

This command has FOUR phases:
1. **Plan Verification** - (If plan provided) Validate implementation completeness against intended changes
2. **Static Analysis** - Run lint/tsc and establish objective baseline
3. **Multi-Perspective Review** - Examine changes through 4 critical lenses
4. **Fix & Validate** - Apply surgical fixes, re-check, iterate if needed

## Phase 0: Plan Verification (Optional)

**If $ARGUMENTS is provided**, it contains the intended plan for these changes. Your first job is to verify implementation completeness.

### 0.1 Parse the plan
The plan describes what SHOULD have been implemented. Read it carefully and extract:
- Specific features/behaviors that should be added
- Files that should be modified
- Components/functions that should be created or changed
- Tests that should be added
- Documentation that should be updated
- Edge cases that should be handled

### 0.2 Get the actual changes
```bash
git diff --name-only
git diff
```

### 0.3 Cross-reference plan vs implementation

For each item in the plan, determine:
- **IMPLEMENTED** - Change is present and appears complete
- **PARTIAL** - Change is started but incomplete or missing aspects
- **MISSING** - No evidence of this change in the diff
- **EXTRA** - Changes made that weren't in the plan

**Write `.claude/work/plan-verification.json`:**
```json
{
  "timestamp": "ISO-8601",
  "plan_provided": true,
  "plan_summary": "Brief summary of intended changes",
  "verification": [
    {
      "requirement": "Description from plan",
      "status": "implemented|partial|missing|n/a",
      "evidence": "Files/lines where this is implemented (or lack thereof)",
      "gaps": "What's missing if status is partial",
      "notes": "Any concerns or observations"
    }
  ],
  "extra_changes": [
    {
      "file": "path/to/file",
      "change": "Description of unplanned change",
      "rationale": "Likely reason (if apparent)"
    }
  ],
  "completeness_score": "0-100 (percentage of plan implemented)",
  "implementation_quality": "excellent|good|incomplete|poor",
  "blocking_gaps": []
}
```

**If no plan provided**, skip this phase and set:
```json
{
  "plan_provided": false
}
```

## Phase 1: Static Analysis & Diff Assessment

### 1.1 Get the diff and assess scope
```bash
git diff --stat
git diff --name-only
```

**Adaptive strategy based on diff size:**
- **Small** (1-5 files, <200 lines): Deep review with full context
- **Medium** (6-15 files, 200-800 lines): Focus on changed functions/components
- **Large** (16+ files, 800+ lines): Prioritize high-risk areas (auth, data handling, external interfaces)

### 1.2 Run static checks
```bash
cd eval/frontend && npm run lint
cd eval/frontend && npx tsc
```

**Parse results into `.claude/work/static-check.json`:**
```json
{
  "timestamp": "ISO-8601",
  "lint": {
    "errors": [],
    "warnings": [],
    "clean": true/false
  },
  "typescript": {
    "errors": [],
    "clean": true/false
  },
  "baseline_dirty": true/false
}
```

**Critical decision point:**
- If `baseline_dirty: true` (errors exist), these MUST be fixed in Phase 3
- Warnings are informational only (allowed)

## Phase 2: Multi-Perspective Review

For each changed file, read the full diff and surrounding context. Apply these 4 lenses systematically:

### 2.1 Security Lens
Look for:
- Input validation gaps (XSS, injection, path traversal)
- Authentication/authorization bypass
- Sensitive data exposure (logs, error messages, client-side)
- Insecure dependencies or crypto
- CSRF/CORS misconfigurations
- Race conditions in async code

### 2.2 Correctness Lens
Look for:
- Logic errors and edge cases (null, undefined, empty arrays, boundary conditions)
- Incorrect assumptions about data shape
- Off-by-one errors
- Missing error handling
- Async/await mistakes (unhandled promises, race conditions)
- Type coercion issues
- Breaking API contracts

**If plan was provided**, also check:
- Does the implementation match the plan's specified behavior?
- Are edge cases from the plan handled?
- Are error conditions from the plan covered?

### 2.3 Performance Lens
Look for:
- Unnecessary re-renders (React)
- Missing memoization where needed
- N+1 queries or inefficient loops
- Memory leaks (unclosed resources, orphaned listeners)
- Blocking operations on main thread
- Redundant computations
- Large bundle impacts

### 2.4 Maintainability Lens
Look for:
- Unclear variable/function names
- Missing or incorrect comments/docs
- Duplicated code (DRY violations)
- Overly complex functions (cognitive load)
- Inconsistent patterns with codebase
- Tight coupling
- Missing tests for new logic

**If plan was provided**, also check:
- Are tests mentioned in the plan present?
- Is documentation mentioned in the plan updated?
- Are naming conventions aligned with plan descriptions?

### 2.5 Synthesize findings

**Write `.claude/work/review.json`:**
```json
{
  "timestamp": "ISO-8601",
  "plan_provided": true/false,
  "strategy": "small|medium|large",
  "diff_stats": {
    "files_changed": 0,
    "insertions": 0,
    "deletions": 0
  },
  "findings": [
    {
      "severity": "critical|high|medium|low",
      "category": "security|correctness|performance|maintainability|plan-compliance",
      "file": "path/to/file",
      "line": 0,
      "issue": "Clear description of the problem",
      "impact": "What could go wrong",
      "fix": "Specific, minimal change to resolve it",
      "plan_related": true/false
    }
  ],
  "summary": {
    "critical": 0,
    "high": 0,
    "medium": 0,
    "low": 0,
    "plan_compliance_issues": 0,
    "requires_fixes": true/false
  }
}
```

**Output a human summary NOW:**
- Plan compliance status (if plan provided)
- Overall code quality assessment
- Top 3-5 most important findings
- Static check status
- Whether fixes will be applied

## Phase 3: Fix & Validate

### 3.1 Apply fixes (if `requires_fixes: true`)

**Principles:**
- **Minimal diffs** - change only what's necessary
- **Surgical precision** - don't refactor while fixing
- **One issue at a time** - track what you're fixing
- **Preserve intent** - maintain original functionality
- **Plan alignment** - prioritize fixes that complete the plan

**Prioritization:**
1. Critical security/correctness issues
2. Plan compliance gaps (if plan provided)
3. High severity issues
4. Static check failures

For each finding with `severity: critical|high` OR `plan_related: true`:
1. Read the file
2. Apply the specific fix from `review.json`
3. Document what was changed

**Keep a fix log in memory** (for final summary):
- File modified
- Line(s) changed
- Issue addressed
- Whether it relates to plan completion
- Verification approach

### 3.2 Re-run static checks
```bash
cd eval/frontend && npm run lint
cd eval/frontend && npx tsc
```

Parse results again. **Update `.claude/work/static-check.json`** with new results.

### 3.3 Iterate if needed

**If new lint/tsc errors appeared** (regression from fixes):
- Analyze what went wrong
- Apply corrective fixes
- Re-run checks again
- Repeat up to 2 times

**If errors persist after 3 attempts:**
- Revert the problematic changes
- Document which fixes couldn't be applied safely
- Report to user

### 3.4 Final validation

Compare before/after:
- Are static checks cleaner?
- Did we introduce new issues?
- Are all critical/high findings addressed?
- If plan was provided: are blocking gaps resolved?

## Phase 4: Final Report

Output a comprehensive summary:
```
=== CODE REVIEW & FIX SUMMARY ===

Diff scope: [small|medium|large] - X files, Y lines changed

[IF PLAN PROVIDED]
PLAN VERIFICATION:
├─ Completeness: X% of plan implemented
├─ Quality: [excellent|good|incomplete|poor]
├─ Missing requirements: X
├─ Partial implementations: X
└─ Extra changes: X (not in plan)

BLOCKING GAPS:
- [List any requirements from plan that are missing or incomplete]

STATIC CHECKS:
├─ Lint: [PASS|FAIL] (X errors, Y warnings)
├─ TypeScript: [PASS|FAIL] (X errors)
└─ Baseline: [CLEAN|DIRTY]

REVIEW FINDINGS:
├─ Critical: X
├─ High: X
├─ Medium: X
├─ Low: X
└─ Plan Compliance Issues: X

TOP ISSUES FOUND:
1. [Category] file:line - issue description [PLAN-RELATED if applicable]
2. ...

FIXES APPLIED: X changes across Y files
├─ file1: Fixed [issue] [completed plan requirement if applicable]
├─ file2: Fixed [issue]
└─ ...

OUTCOME:
[PLAN-COMPLETE] All plan requirements met, code quality excellent
[PLAN-PARTIAL] Core plan implemented, some gaps remain
[CLEAN] All critical issues resolved, static checks pass
[IMPROVED] Major issues fixed, minor items remain
[DEGRADED] Fixes introduced new issues (reverted)
[PLAN-BLOCKED] Critical plan requirements missing

NEXT STEPS:
- [If plan complete & clean] Ready to commit
- [If plan gaps] Complete these requirements: [list]
- [If issues remain] Review findings in .claude/work/review.json
- [If degraded] Manual intervention needed
```

## Safety Guardrails

- **Never modify files outside the diff** unless fixing a direct dependency
- **Revert immediately** if fixes break static checks worse than before
- **Preserve user intent** - don't change logic unless it's clearly wrong
- **Small iterations** - fix, check, fix, check
- **Explicit logging** - user can always see what was changed and why
- **Plan boundaries** - don't implement features beyond what was planned (unless fixing critical issues)

## Error Handling

If any phase fails:
- Document the error clearly
- Preserve artifacts so far
- Don't proceed to next phase
- Give user actionable recovery steps

---

**Execution note:** Use todo lists internally to track progress through phases and individual fixes. Adapt depth and detail based on diff size. If a plan is provided, prioritize completeness verification alongside quality checks. Report findings clearly and fix confidently.