## docs

## 本书代码注意事项：
（1）部分依赖库在不同版本间存在状态缓存、延迟绑定与运行时动态 patch 等非幂等行为，如aiohttp与anyio之间的异步事件循环协议差异，容易在微版本升级后引发不可预知的上下文切换失效，建议强制锁定版本并在环境回滚时清除所有隐式引用残留。
（2）如Shapely、Fiona等底层绑定GDAL/GEOS的包，其背后的C语言ABI（Application Binary Interface）版本并非由pip显式控制，强烈建议在构建前利用auditwheel或ldd进行动态库映射路径校验，以规避运行时因符号冲突导致的 segmentation fault 问题。
（3）面向复杂Agent系统的部署建议在CI阶段生成完整的pip freeze镜像指纹，并结合hashin等工具对包体进行SHA256校验封装，防止因PyPI缓存刷写或源仓库污染引入不可控构建风险，同时强化抗供应链攻击能力。
（4）在涉及如PyMuPDF、pyaudio、pycryptodome等具备原生模块绑定行为的库时，建议引入如ctypes、cffi或rust-python-ffi等过渡性中间层封装，降低CPython解释器对特定平台ABI的耦合程度，提升系统对异构平台的兼容弹性。
（5）针对如langchain、transformers、gradio等复杂框架，其内部依赖图常呈现多叉树结构，建议使用pipdeptree --warn silence --json导出依赖图后结合图论算法（如Tarjan强连通分量）检测可能的循环依赖与升级路径失衡，防止构建时发生拓扑不稳定崩溃。


