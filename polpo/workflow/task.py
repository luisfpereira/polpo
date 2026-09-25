import logging
import platform
from abc import ABC
from pathlib import Path

from polpo.io.json import load_json, save_json
from polpo.time import Timer, utc_now


def _get_default_logger(runner):
    return logging.getLogger(type(runner).__module__)


def task(func):
    """Mark a method as a runnable task.

    Parameters
    ----------
    func : callable
        Method to mark as a task.

    Returns
    -------
    func : callable
        The marked method.
    """
    func._is_task = True
    return func


class TaskRunner(ABC):
    """Run named tasks with persistent state and failure tracking.

    Tasks are methods decorated with :func:`task`. Their execution state is
    stored in a manifest, allowing completed tasks to be skipped on subsequent
    runs.

    Parameters
    ----------
    state_dir : path-like
        Directory where the runner state and manifest are stored.
    metadata : dict
        Metadata to include in the manifest.
    logger : logging.Logger
        Logger used to report execution progress.
    """

    MANIFEST_VERSION = 1

    def __init__(self, state_dir, metadata=None, logger=None):
        self.state_dir = state_dir
        self.metadata = metadata or None
        self.resolved_ = {}
        self.timer = Timer()
        self.logger = logger or _get_default_logger(self)

    def from_callable(cls, fn, state_dir=None, name=None, metadata=None, logger=None):
        """Create a task runner from a callable.

        Parameters
        ----------
        fn : callable
            Callable to execute as the runner's single task.
        state_dir : path-like
            Directory where the runner state and manifest are stored.
        name : str or None
            Task name. If None, ``fn.__name__`` is used.
        metadata : dict or None
            Metadata to include in the manifest.
        logger : logging.Logger or None
            Logger used to report execution progress.

        Returns
        -------
        runner : SingleTaskRunner
            Task runner wrapping the callable.
        """
        return SingleTaskRunner(
            fn, state_dir, name=name, metadata=metadata, logger=logger
        )

    @property
    def manifest_path(self):
        """Path to the runner manifest."""
        return self.state_dir / "manifest.json"

    def tasks(self):
        """Return the available tasks.

        Tasks are collected from methods decorated with :func:`task`, including
        tasks inherited from base classes.

        Returns
        -------
        tasks : dict
            Mapping from task names to bound task methods.
        """
        tasks = {}

        for cls in reversed(type(self).mro()):
            for name, attr in cls.__dict__.items():
                if getattr(attr, "_is_task", False):
                    tasks[name] = getattr(self, name)

        return tasks

    def set_resolved(self, **values):
        """Store resolved values in the runner state.

        Parameters
        ----------
        **values
            Values to include in the manifest under ``resolved``.
        """
        self.resolved_.update(values)

    def _new_manifest(self):
        """Create a new runner manifest."""
        return {
            "version": self.MANIFEST_VERSION,
            "runner": f"{type(self).__module__}.{type(self).__qualname__}",
            "machine": platform.node(),
            "metadata": self.metadata,
            "resolved": self.resolved_,
            "status": "running",
            "started_at": utc_now(),
            "tasks": {},
        }

    def _load_or_create_manifest(self):
        """Load the existing manifest or create a new one."""
        if self.manifest_path.exists():
            return load_json(self.manifest_path)

        return self._new_manifest()

    def _write_manifest(self):
        """Write the current runner state to the manifest."""
        self.manifest_["resolved"] = dict(self.resolved_)
        self.manifest_["updated_at"] = utc_now()
        save_json(self.manifest_path, self.manifest_)

    def _is_complete(self, task):
        """Return whether a task is marked as completed."""
        task_info = self.manifest_["tasks"].get(task, {})
        return task_info.get("status") == "completed"

    def _mark_completed(self, task):
        """Mark a task as completed in the manifest."""
        self.manifest_["tasks"][task] = {
            "status": "completed",
            "finished_at": utc_now(),
            "elapsed": self.timer.as_dict()[task],
        }

    def _mark_failed(self, task, error):
        """Mark a task as failed in the manifest."""
        self.manifest_["status"] = "failed"
        self.manifest_["tasks"][task] = {
            "status": "failed",
            "finished_at": utc_now(),
            "error": {
                "type": type(error).__name__,
                "message": str(error),
            },
        }

    def _resolve_tasks(self, tasks, exclude_tasks):
        """Resolve and validate the tasks selected for execution.

        Parameters
        ----------
        tasks : sequence of str or None
            Names of tasks to run. If None, all available tasks are selected.
        exclude_tasks : sequence of str or None
            Names of tasks to exclude from execution.

        Returns
        -------
        tasks : dict
            Mapping from selected task names to tasks.

        Raises
        ------
        ValueError
            If a requested or excluded task is not available.
        """
        available_tasks = self.tasks()

        if tasks is None:
            tasks = list(available_tasks)

        exclude_tasks = [] if exclude_tasks is None else exclude_tasks

        unknown = (set(tasks) | set(exclude_tasks)) - set(available_tasks)
        if unknown:
            raise ValueError(f"Unknown tasks: {sorted(unknown)}")

        return {
            task: available_tasks[task] for task in tasks if task not in exclude_tasks
        }

    def _run_tasks(self, tasks, overwrite, continue_on_error):
        """Run resolved tasks and update their execution state.

        Parameters
        ----------
        tasks : dict
            Mapping from task names to callable tasks.
        overwrite : bool
            Whether to rerun tasks already marked as completed.
        continue_on_error : bool
            Whether to continue running tasks after a task fails.

        Returns
        -------
        runner : TaskRunner
            This runner.
        """
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_ = self._load_or_create_manifest()

        self.logger.info("Running %s", self.__class__.__name__)
        self.logger.info("State directory: %s", self.state_dir)

        n_failed = 0
        for task_name, task in tasks.items():
            if not overwrite and self._is_complete(task_name):
                self.logger.info("[%s] skipped (already completed)", task_name)
                continue

            self.logger.info("[%s] started", task_name)

            try:
                with self.timer(task_name):
                    task()
            except Exception as error:
                n_failed += 1
                self._mark_failed(task_name, error)
                self._write_manifest()

                self.logger.error(
                    "[%s] failed: %s",
                    task_name,
                    error,
                    exc_info=self.logger.isEnabledFor(logging.DEBUG),
                )

                if not continue_on_error:
                    raise

                continue

            elapsed = self.timer.duration(task_name)

            self._mark_completed(task_name)
            self._write_manifest()

            self.logger.info(
                "[%s] completed in %.2f s",
                task_name,
                elapsed,
            )

        if n_failed == 0:
            status = "completed"
        elif n_failed == len(tasks):
            status = "failed"
        else:
            status = "partial"

        self.manifest_["status"] = status
        self.manifest_["finished_at"] = utc_now()
        self._write_manifest()

        self.logger.info(
            "%s finished: %s",
            self.__class__.__name__,
            status,
        )

        return self

    def reset(self):
        """Reset the runner execution state.

        Removes the persisted manifest and clears runtime state. Result artifacts
        produced by tasks are left unchanged.

        Returns
        -------
        runner : TaskRunner
            This runner.
        """
        self.manifest_path.unlink(missing_ok=True)

        self.resolved_ = {}
        self.timer = Timer()

        return self

    def run(
        self,
        tasks=None,
        exclude_tasks=None,
        overwrite=False,
        continue_on_error=True,
    ):
        """Run selected tasks.

        Parameters
        ----------
        tasks : sequence of str
            Names of tasks to run. By default, all available tasks are run.
        exclude_tasks : sequence of str
            Names of tasks to exclude.
        overwrite : bool
            Whether to rerun tasks already marked as completed.
        continue_on_error : bool
            Whether to continue running remaining tasks after a task fails.

        Returns
        -------
        runner : TaskRunner
            This runner.
        """
        tasks = self._resolve_tasks(tasks, exclude_tasks)
        return self._run_tasks(tasks, overwrite, continue_on_error)


class CompositeTaskRunner(TaskRunner):
    """Run a collection of task runners as tasks of a parent runner.

    Each child runner is represented as a task of the composite runner.

    A child runner is considered successful only if its final manifest status
    is ``"completed"``.

    Parameters
    ----------
    runners : dict
        Mapping from task names to :class:`TaskRunner` instances.
    state_dir : path-like
        Directory where the composite runner state and manifest are stored.
    metadata : dict
        Metadata to include in the composite manifest.
    logger : logging.Logger
        Logger used to report execution progress.
    """

    def __init__(self, runners, state_dir, metadata=None, logger=None):
        self.runners = runners
        super().__init__(state_dir, metadata=metadata, logger=logger)

        for name, runner in self.runners.items():
            if runner.logger == _get_default_logger(runner):
                runner.logger = self.logger.getChild(name)

    def tasks(self):
        """Return the child runners.

        Returns
        -------
        tasks : dict
            Mapping from task names to child runners.
        """
        return self.runners

    def _make_runner_task(
        self,
        runner,
        overwrite=False,
        continue_on_error=False,
    ):
        """Create a task that executes a child runner.

        Parameters
        ----------
        runner : TaskRunner
            Child runner to execute.
        overwrite : bool
            Whether to rerun completed tasks in the child runner.
        continue_on_error : bool
            Whether the child runner should continue after a task fails.

        Returns
        -------
        task : callable
            Task that executes the child runner.

        Raises
        ------
        RuntimeError
            If the child runner does not finish with status ``"completed"``.
        """

        def run():
            runner.run(
                overwrite=overwrite,
                continue_on_error=continue_on_error,
            )

            if runner.manifest_["status"] != "completed":
                raise RuntimeError(
                    f"{runner.__class__.__name__} finished with "
                    f"status {runner.manifest_['status']!r}"
                )

        return run

    def reset(self):
        """Reset this runner and all child runners."""
        super().reset()

        for runner in self.runners.values():
            runner.reset()

        return self

    def run(
        self,
        tasks=None,
        exclude_tasks=None,
        overwrite=False,
        continue_on_error=True,
    ):
        """Run selected child runners.

        Parameters
        ----------
        tasks : sequence of str
            Names of child runners to run. By default, all are run.
        exclude_tasks : sequence of str
            Names of child runners to exclude.
        overwrite : bool
            Whether to rerun completed tasks in both the composite and child
            runners.
        continue_on_error : bool
            Whether to continue after failures, both within child runners and
            across child runners.

        Returns
        -------
        runner : CompositeTaskRunner
            This runner.
        """
        tasks = self._resolve_tasks(tasks, exclude_tasks)

        tasks = {
            name: self._make_runner_task(
                runner,
                overwrite=overwrite,
                continue_on_error=continue_on_error,
            )
            for name, runner in tasks.items()
        }

        return self._run_tasks(
            tasks,
            overwrite=overwrite,
            continue_on_error=continue_on_error,
        )


class SingleTaskRunner(TaskRunner):
    """Run a single callable as a named task.

    Parameters
    ----------
    fn : callable
        Callable to execute.
    state_dir : path-like
        Directory where the runner state and manifest are stored.
    name : str or None
        Task name. If None, ``fn.__name__`` is used.
    metadata : dict or None
        Metadata to include in the manifest.
    logger : logging.Logger or None
        Logger used to report execution progress.
    """

    def __init__(self, fn, state_dir=None, name=None, metadata=None, logger=None):
        if name is None:
            name = fn.__name__

        if state_dir is None:
            state_dir = Path(f".{name}")

        super().__init__(state_dir, metadata, logger)

        self.fn = fn
        self.name = name

    def tasks(self):
        """Return the single task.

        Returns
        -------
        tasks : dict
            Mapping from the task name to the wrapped callable.
        """
        return {self.name: self.fn}
