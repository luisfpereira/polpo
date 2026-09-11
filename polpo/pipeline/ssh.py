import os

import paramiko
import scp

from polpo.defaults import DATA_DIR

from .base import CacheableDataLoader, PreprocessingStep


class SshClient:
    def __init__(self, host_name="frank"):
        self.host_name = host_name
        self._client = None

    def connect(self):
        """Connect to a host using SSH config."""
        config = paramiko.SSHConfig()
        with open(os.path.expanduser("~/.ssh/config")) as file:
            config.parse(file)

        host_config = config.lookup(self.host_name)

        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        client.connect(
            hostname=host_config.get("hostname", self.host_name),
            port=host_config.get("port", 22),
            username=host_config.get("user"),
            key_filename=host_config.get("identityfile"),
        )

        self._client = client

        return self

    def to_scp(self):
        if self._client is None:
            self.connect()

        return scp.SCPClient(self._client.get_transport())


class ScpDataLoader(PreprocessingStep, CacheableDataLoader):
    """Transfer files and directories from remote host to localhost.

    Parameters
    ----------
    remote_path : str
        Path to retrieve from remote host.
    ssh_client : SshClient
        Client handling connection.
    data_dir : str
        Directory where to store data.
    recursive : str
        Whether to transfer files and directories recursively.
    use_cache : bool
        Whether to verify if data is already available locally.
    local_basename : str
        Basename of transferred file/folder if different from remote host.
    """

    def __init__(
        self,
        remote_path,
        ssh_client,
        data_dir=None,
        recursive=True,
        use_cache=True,
        local_basename=None,
    ):
        super().__init__(use_cache)

        if data_dir is None:
            data_dir = DATA_DIR

        if "~" in data_dir:
            data_dir = os.path.expanduser(data_dir)

        self.ssh_client = ssh_client
        self.remote_path = remote_path
        self.recursive = recursive
        self.data_dir = data_dir
        self.local_basename = local_basename

    @property
    def _path(self):
        remote_path_ls = self.remote_path.split(os.path.sep)
        return os.path.join(self.data_dir, remote_path_ls[-1])

    @property
    def _renamed_path(self):
        if self.local_basename is None:
            return self._path

        return os.path.join(self.data_dir, self.local_basename)

    @classmethod
    def from_host_name(
        cls,
        host_name,
        remote_path,
        data_dir=None,
        recursive=True,
        use_cache=True,
        local_basename=None,
        force_connect=False,
    ):
        ssh_client = SshClient(host_name)
        # allows to detect inability to connect at instantiation
        if force_connect:
            ssh_client.connect()

        return cls(
            remote_path,
            ssh_client=ssh_client,
            data_dir=data_dir,
            recursive=recursive,
            use_cache=use_cache,
            local_basename=local_basename,
        )

    def load(self):
        path = self._renamed_path

        if self.exists(path):
            return path

        os.makedirs(self.data_dir, exist_ok=True)
        with self.ssh_client.to_scp() as scp_client:
            scp_client.get(
                remote_path=self.remote_path,
                local_path=self.data_dir,
                recursive=self.recursive,
            )

        if self.local_basename and self._path != self._renamed_path:
            os.rename(self._path, self._renamed_path)

        return path

    def __call__(self, data=None):
        return self.load()
