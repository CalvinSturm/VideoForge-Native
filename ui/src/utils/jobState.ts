import type { Job, ParsedRaveError } from "../types";

type JobPatch = Partial<Job>;

export function completeJob(job: Job, patch: JobPatch = {}): Job {
    return {
        ...job,
        status: "done",
        progress: 100,
        eta: 0,
        completedAt: Date.now(),
        ...patch,
    };
}

export function failJob(
    job: Job,
    error: ParsedRaveError,
    patch: JobPatch = {}
): Job {
    return {
        ...job,
        status: "error",
        statusMessage: `[${error.category}] ${error.message}`,
        errorCategory: error.category,
        ...(error.nextAction ? { errorHint: error.nextAction } : {}),
        errorMessage: error.detail ? `${error.message} :: ${error.detail}` : error.message,
        completedAt: Date.now(),
        ...patch,
    };
}

export function cancelJob(job: Job, patch: JobPatch = {}): Job {
    return {
        ...job,
        status: "cancelled",
        progress: 0,
        eta: 0,
        completedAt: Date.now(),
        ...patch,
    };
}

export function updateJobById(
    jobs: Job[],
    jobId: string,
    updater: (job: Job) => Job
): Job[] {
    return jobs.map(job => (job.id === jobId ? updater(job) : job));
}

export function updateJobs(
    jobs: Job[],
    predicate: (job: Job) => boolean,
    updater: (job: Job) => Job
): Job[] {
    return jobs.map(job => (predicate(job) ? updater(job) : job));
}

export function updateActiveJobById(
    activeJob: Job | null,
    jobId: string,
    updater: (job: Job) => Job
): Job | null {
    if (!activeJob || activeJob.id !== jobId) {
        return activeJob;
    }

    return updater(activeJob);
}

export function updateActiveJob(
    activeJob: Job | null,
    predicate: (job: Job) => boolean,
    updater: (job: Job) => Job
): Job | null {
    if (!activeJob || !predicate(activeJob)) {
        return activeJob;
    }

    return updater(activeJob);
}
