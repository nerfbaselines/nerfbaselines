import functools
from nerfbaselines.types import Method, MethodInfo, ModelInfo


def _wrap_server_result(fn):
    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        try:
            return True, fn(self, *args, **kwargs)
        except Exception as e:
            return False, e
    return wrapper


class RPCServerBase:
    def __init__(self, inqueue, outqueue):
        self._inqueue = inqueue
        self._outqueue = outqueue

    def __new__(cls, *args, **kwargs):
        del args, kwargs
        self = super().__new__(cls)
        for name in dir(cls):
            if name.startswith("method_"):
                setattr(self, name, _wrap_server_result(getattr(self, name)))
        return self

    def _handle_message(self, message):
        method_name, args, kwargs = message
        method = getattr(self, method_name)
        try:
            result = True, method(*args, **kwargs)
        except Exception as e:
            result = False, e
        self._outqueue.put(result)


class RPCServer(RPCServerBase, implements=(Method,)):
    pass


class RPCServerProtocol:
    def method_init(self, method_class) -> int:
        ...
        
    def method_get_info_static(self, method_class) -> MethodInfo:
        ...

    def method_install_static(self, method_class) -> None:
        ...

    def method_get_info(self, instance) -> ModelInfo:
        ...

    def method_render_start(self, instance, *args, **kwargs) -> None:
        ...

    def method_render_next(self, instance) -> RenderOutput:
        ...

    def method_render_end(self, render_instance) -> None:
        ...

    def method_del(self, instance) -> None:
        ...

    def method_get_train_embedding(self, instance, *args, **kwargs) -> TrainEmbeddingOutput:
        ...

    def method_optimize_embeddings_start(self, instance, *args, **kwargs) -> OptimizeEmbeddingsOutput:
        ...

    def method_optimize_embeddings_next(self, optimize_instance) -> OptimizeEmbeddingsOutput:
        ...

    def method_optimize_embeddings_end(self, optimize_instance) -> None:
        ...

    def method_train_iteration(self, instance, *args, **kwargs) -> TrainIterationOutput:
        ...

    def method_save(self, instance, *args, **kwargs) -> None:
        ...


class RPCServer(RPCServerProtocol):
    def __init__(self):
        self._methods = {}
        self._render_instances = {}

    def method_init(self, method_class, *args, **kwargs):
        instance_id = len(self._methods)
        self._methods[instance_id] = method_class(*args, **kwargs)
        return instance_id

    def method_get_info_static(self, method_class):
        return method_class.get_method_info()


def _auto_rpc_method_proxy(fn):
    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        self._outqueue.put(fn.__name__, *args, **kwargs)
        return self._unwrap_queue_item(self._inqueue.get())
    return wrapper


class RPCServerProxy(RPCServerProtocol):
    def __init__(self, inqueue, outqueue):
        self._inqueue = inqueue
        self._outqueue = outqueue
        self._render_instances = {}

    def _unwrap_queue_item(self, item):
        success, result = item
        if success:
            return self._unwrap_queue_item(result)
        else:
            raise result
        
# Auto-implement all methods from RPCServerProtocol
for name in dir(RPCServerProtocol):
    if not hasattr(RPCServerProxy, name):
        setattr(RPCServerProxy, name, _auto_rpc_method_proxy(getattr(RPCServerProtocol, name)))


class RPCServer:
    def __init__(self):
        self._methods = {}
        self._render_instances = {}
        
    def _resolve_method_class(self, method_class: str) -> Type[Method]:
        return ...

    def method_init(self, method_class, *args, **kwargs):
        method = method_class(*args, **kwargs)
        self._methods[id(method)] = method
        return id(method)

    def method_get_info_static(self, method_class):
        ...


class RPCMethod(Method):
    _server: RPCServer
    _method_class: str

    def __init__(self, *args, **kwargs):
        self._instance_id = self._server.method_init(self._method_class, *args, **kwargs)

    @classmethod
    def install(cls):
        cls._server.method_install_static(cls._method_class)

    @classmethod
    def get_method_info(cls) -> MethodInfo:
        cls._server.method_get_info_static()

    def __del__(self):
        self._server.method_del(self._instance_id)

    def get_info(self) -> ModelInfo:
        return self._server.method_get_info(self._instance_id)

    def get_train_embedding(self, *args, **kwargs):
        return self._server.method_get_train_embedding(self._instance_id, *args, **kwargs)

    def optimize_embeddings(self, *args, **kwargs):
        optimize_instance = self._server.method_optimize_embeddings_start(self._instance_id, *args, **kwargs)

        class OptimizeIterator:
            def __init__(self, optimize_instance):
                self._optimize_instance = optimize_instance

            def __iter__(self):
                return self

            def __next__(self):
                return self._server.method_optimize_embeddings_next(self._optimize_instance)

            def __del__(self):
                self._server.method_optimize_embeddings_end(self._optimize_instance)

        return OptimizeIterator(optimize_instance)

    def render(self, *args, **kwargs):
        render_instance = self._server.method_render_start(self._instance_id, *args, **kwargs)

        class RenderIterator:
            def __init__(self, render_instance):
                self._render_instance = render_instance

            def __iter__(self):
                return self

            def __next__(self):
                return self._server.method_render_next(self._render_instance)

            def __del__(self):
                self._server.method_render_end(self._render_instance)

        return RenderIterator(render_instance)

    def train_iteration(self, *args, **kwargs):
        return self._server.method_train_iteration(self._instance_id, *args, **kwargs)

    def save(self, *args, **kwargs):
        return self._server.method_save(self._instance_id, *args, **kwargs)
