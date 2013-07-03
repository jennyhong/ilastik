import ConfigParser
import io, os

default_config = """
[ilastik]
debug: false
plugin_directories: ~/.ilastik/plugins,
"""

cfg = ConfigParser.SafeConfigParser()
cfg.readfp(io.BytesIO(default_config))
userConfig = os.path.expanduser("~/.ilastikrc")
if os.path.exists(userConfig):
    cfg.read(userConfig)