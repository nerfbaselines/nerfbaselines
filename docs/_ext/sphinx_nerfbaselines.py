from typing import TYPE_CHECKING
import sphinx.application
from docutils import nodes

from docutils.parsers.rst import directives
from sphinx.util.docutils import SphinxDirective
from sphinx.application import Sphinx
from sphinx.directives.code import CodeBlock

if TYPE_CHECKING:
    from nerfbaselines import MethodSpec


class NerfBaselinesDirective(SphinxDirective):
    """A directive to render a method."""
    has_content = False
    optional_arguments = 1
    option_spec = {
        'names-wildcard': directives.unchanged,
        'names-regex': directives.unchanged,
        'long-ids': directives.flag,
    }

    def _get_all_objects(self):
        self._register_all()

        try:
            import nerfbaselines._registry as registry
        except ImportError:
            # Older versions
            import nerfbaselines.registry as registry  # type: ignore
        try:
            methods_registry = registry.methods_registry
        except AttributeError:
            # Older versions
            methods_registry = registry.registry
        for name, spec in methods_registry.items():
            metadata = getattr(spec, "metadata", None)
            if metadata is None:
                try:
                    metadata = spec.get("metadata")
                except AttributeError:
                    pass
            if metadata is None:
                continue
            yield "methods/" + name, spec

        try:
            for name, spec in registry.datasets_registry.items():
                yield "datasets/" + name, spec
        except AttributeError:
            pass

        try:
            for name, spec in registry.evaluation_protocols_registry.items():
                yield "evaluation-protocols/" + name, spec
        except AttributeError:
            pass

        try:
            for name, spec in registry.loggers_registry.items():
                yield "loggers/" + name, spec
        except AttributeError:
            pass

    def _render_method(self, name, spec: 'MethodSpec'):
        # If short-ids is not set, remove the prefix
        if 'long-ids' not in self.options:
            name = name.split("/", 1)[-1]

        from nerfbaselines import backends
        meta = getattr(spec, "metadata", None)
        if meta is None:
            try:
                meta = spec.get("metadata")
            except AttributeError:
                pass
        assert meta is not None, f"Method {name} has no metadata"
        section = nodes.section(
            '',
            nodes.title(text=meta["name"]),
            ids=[name],
            names=[nodes.fully_normalize_name(name)],
        )
        if meta.get("paper_title"):
            section += nodes.rubric("", nodes.Text(meta["paper_title"]))

        fields = []
        if meta.get("paper_authors"):
            fields.append(nodes.field('', 
                                      nodes.field_name('', nodes.Text("Authors")), 
                                      nodes.field_body('', nodes.Text(', '.join(meta.get("paper_authors"))))))
        if meta.get("paper_link"):
            link = nodes.paragraph('')
            link += nodes.reference('', nodes.Text(meta["paper_link"]), internal=False, refuri=meta["paper_link"])
            fields.append(nodes.field('', 
                                      nodes.field_name('', nodes.Text("Paper")), 
                                      nodes.field_body('', link)))
        if meta.get("link"):
            link = nodes.paragraph('')
            link += nodes.reference('', nodes.Text(meta["link"]), internal=False, refuri=meta["link"])
            fields.append(nodes.field('', 
                                      nodes.field_name('', nodes.Text("Web")), 
                                      nodes.field_body('', link)))

        if meta.get("licenses"):
            licenses = nodes.paragraph('')
            for i, li in enumerate(meta.get("licenses")):
                if isinstance(li, str):
                    li = {"name": li}
                if i != 0:
                    licenses += nodes.Text(", ")
                if li.get("url"):
                    licenses += nodes.reference('', nodes.Text(li["name"]), internal=False, refuri=li["url"])
                else:
                    licenses += nodes.Text(li["name"])
            fields.append(nodes.field('', 
                                      nodes.field_name('', nodes.Text("Licenses")), 
                                      nodes.field_body('', licenses)))

        fields.append(nodes.field('', 
                                  nodes.field_name('', nodes.Text("ID")), 
                                  nodes.field_body('', nodes.Text(name))))

        try:
            supported_backends = backends.get_implemented_backends(spec)
            fields.append(nodes.field('', 
                                      nodes.field_name('', nodes.Text("Backends")), 
                                      nodes.field_body('', nodes.Text(", ".join(supported_backends)))))
        except AttributeError:
            pass

        info = {}
        try:
            from nerfbaselines.results import get_method_info_from_spec
            info = get_method_info_from_spec(spec)
        except ImportError:
            pass
        def _get_field_value(strings):
            field_value = nodes.container()
            for i, string in enumerate(strings):
                if i > 0:
                    field_value += nodes.Text(", ")
                field_value += nodes.literal('', nodes.Text(string))
            return field_value

        if info.get("supported_camera_models"):
            fields.append(nodes.field('', 
                                      nodes.field_name('', nodes.Text("Camera models")), 
                                      nodes.field_body('', _get_field_value(info.get("supported_camera_models", [])))))
        if info.get("required_features"):
            # Use '`{text}`' nodes separated by ', ' string
            fields.append(nodes.field('', 
                                      nodes.field_name('', nodes.Text("Required features")), 
                                      nodes.field_body('', _get_field_value(info.get("required_features", [])))))
        if info.get("supported_outputs"):
            supported_outputs = [x if isinstance(x, str) else x["name"] for x in info.get("supported_outputs", [])]
            fields.append(nodes.field('', 
                                      nodes.field_name('', nodes.Text("Supported outputs")), 
                                      nodes.field_body('', _get_field_value(supported_outputs))))

        section += nodes.field_list('', *fields)
        # section += nodes.paragraph("", nodes.Text(meta["description"]))
        if meta.get("description"):
            section += self.parse_text_to_nodes(meta["description"])
        if meta.get("long_description"):
            section += self.parse_text_to_nodes(meta["long_description"], allow_section_headings=True)
        return section

    def _register_all(self):
        try:
            import nerfbaselines._registry as registry
        except ImportError:
            # Older versions
            import nerfbaselines.registry as registry  # type: ignore
        # Calls register_all
        try:
            registry.get_supported_methods()
        except:
            # Older versions of nerfbaselines
            registry.supported_methods()

    def _resolve_evaluation_protocol(self, name):
        try:
            import nerfbaselines._registry as registry
        except ImportError:
            # Older versions
            import nerfbaselines.registry as registry  # type: ignore
        self._register_all()
        resolve_target = getattr(self.env.config, 'linkcode_resolve', None)
        spec = registry.evaluation_protocols_registry[name]
        name = spec.get("evaluation_protocol_class", spec.get("evaluation_protocol"))
        module, fullname = name.split(":", 1)
        return resolve_target("py", {"module": module, "fullname": fullname})

    def _render_dataset(self, name, spec):
        # If long-ids is not set, remove the prefix
        qualname = name
        name = name.split("/", 1)[-1]
        if 'long-ids' not in self.options:
            qualname = name

        meta = spec.get("metadata", {})
        if not "name" in meta:
            # Pure dataloader
            return None
        name = meta.get("name", name)
        section = nodes.section(
            '',
            nodes.title(text=name),
            ids=[qualname],
            names=[nodes.fully_normalize_name(qualname)],
        )
        if meta.get("paper_title"):
            section += nodes.rubric("", nodes.Text(meta["paper_title"]))

        fields = []
        if meta.get("paper_authors"):
            fields.append(nodes.field('', 
                                      nodes.field_name('', nodes.Text("Authors")), 
                                      nodes.field_body('', nodes.Text(', '.join(meta.get("paper_authors"))))))
        if meta.get("paper_link"):
            link = nodes.paragraph('')
            link += nodes.reference('', nodes.Text(meta["paper_link"]), internal=False, refuri=meta["paper_link"])
            fields.append(nodes.field('', 
                                      nodes.field_name('', nodes.Text("Paper")), 
                                      nodes.field_body('', link)))
        if meta.get("link"):
            link = nodes.paragraph('')
            link += nodes.reference('', nodes.Text(meta["link"]), internal=False, refuri=meta["link"])
            fields.append(nodes.field('', 
                                      nodes.field_name('', nodes.Text("Web")), 
                                      nodes.field_body('', link)))

        if meta.get("licenses"):
            licenses = nodes.paragraph('')
            for i, li in enumerate(meta.get("licenses")):
                if isinstance(li, str):
                    li = {"name": li}
                if i != 0:
                    licenses += nodes.Text(", ")
                if li.get("url"):
                    licenses += nodes.reference('', nodes.Text(li["name"]), internal=False, refuri=li["url"])
                else:
                    licenses += nodes.Text(li["name"])
            fields.append(nodes.field('', 
                                      nodes.field_name('', nodes.Text("Licenses")), 
                                      nodes.field_body('', licenses)))

        fields.append(nodes.field('', 
                                  nodes.field_name('', nodes.Text("ID")), 
                                  nodes.field_body('', nodes.Text(name))))

        evaluation_protocol = spec.get("evaluation_protocol") or "default"
        evaluation_protocol_target = self._resolve_evaluation_protocol(evaluation_protocol)
        link = nodes.paragraph('')
        link += nodes.reference('', nodes.Text(evaluation_protocol + " (source code)"), internal=False, refuri=evaluation_protocol_target)
        fields.append(nodes.field('', 
                                  nodes.field_name('', nodes.Text("Evaluation protocol")),
                                  nodes.field_body('', link)))

        section += nodes.field_list('', *fields)
        section += nodes.paragraph("", nodes.Text(meta["description"]) if meta.get("description") else "")
        return section

    def _render_evaluation_protocol(self, name, spec):
        section = nodes.section(
            '',
            nodes.title(text=name),
            ids=[name],
            names=[nodes.fully_normalize_name(name)],
        )
        return section

    def _render_logger(self, name, spec):
        section = nodes.section(
            '',
            nodes.title(text=name),
            ids=[name],
            names=[nodes.fully_normalize_name(name)],
        )
        return section

    def run(self) -> list[nodes.Node]:
        names = []
        supported_objects = dict(self._get_all_objects())
        if len(self.arguments) == 1:
            names = [self.arguments[0]]
        elif 'names-wildcard' in self.options:
            # Wildcard matching
            import fnmatch
            for name in supported_objects:
                if fnmatch.fnmatch(name, self.options['names-wildcard']):
                    names.append(name)
        elif 'names-regex' in self.options:
            import re
            for name in supported_objects:
                if re.match(self.options['names-regex'], name):
                    names.append(name)
        else:
            names = supported_objects.keys()

        sections = []
        names.sort(key=lambda x: x.split("/", 1)[-1].lower())
        for name in names:
            spec = supported_objects[name]
            if name.startswith("methods/"):
                section = self._render_method(name, spec)
            elif name.startswith("datasets/"):
                section = self._render_dataset(name, spec)
            elif name.startswith("evaluation-protocols/"):
                section = self._render_evaluation_protocol(name, spec)
            elif name.startswith("loggers/"):
                section = self._render_logger(name, spec)
            else:
                raise ValueError(f"Unknown object type {name}")
            if section is not None:
                sections.append(section)
        return sections


class NerfBaselinesInstallBlock(CodeBlock):
    has_content = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.arguments = ['bash']
        html_context = self.state.document.settings.env.config["html_context"]
        versions = html_context["versions"].split(",")
        if len(versions) < 2:
            # Build --docs latest
            self.content = ['pip install nerfbaselines']
            return
        version_names = html_context["version_names"].split(",")
        current_version = html_context["current_version"]
        current_version_name = version_names[versions.index(current_version)]
        if current_version_name == "latest":
            self.content = ['pip install nerfbaselines']
        elif current_version_name == "dev":
            try:
                from nerfbaselines._constants import CODE_REPOSITORY
            except ImportError:
                CODE_REPOSITORY = "github.com/nerfbaselines/nerfbaselines"
            self.content = [f'pip install git+https://{CODE_REPOSITORY}.git']
        else:
            self.content = [f'pip install nerfbaselines=={html_context["current_version"]}']


def setup(app: sphinx.application.Sphinx):
    app.add_directive('nerfbaselines', NerfBaselinesDirective)
    app.add_directive('nerfbaselines-install', NerfBaselinesInstallBlock)

    return {
        'version': '0.1',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
