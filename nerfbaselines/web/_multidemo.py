"""
Helper functions for generating multi-demo - a demo with a toggle to switch between different configurations.
"""
import os


_css = """
#_multidemo {
  display: flex;
  justify-content: center;
  align-items: center;
  position: absolute;
  left: 50%;
  transform: translateX(-50%);
  z-index: 100000000;
  top: 1rem;
}
#_multidemo label {
  font-weight: 600;
  font-family: sans-serif;
  cursor: pointer;
  padding: 10px 20px;
  background-color: #fff;
  border: 2px solid #fff;
  margin-right: -1px;
  transition: background-color 0.3s;
}
#_multidemo input[type="radio"] { display: none; }
#_multidemo input[type="radio"]:checked + label {
  background-color: rgb(59 130 246);
  color: white;
}
#_multidemo input[type="radio"]:not(:checked) + label { background-color: #f0f0f0; }
#_multidemo label:last-of-type { border-radius: 0 1em 1em 0; }
#_multidemo label:first-of-type { border-radius: 1em 0 0 1em; }
"""

_js = """
const params=new URLSearchParams(window.location.search);
const defaultParams = params.get('p') || 'params.json';
if (params.get('p0')) {
    console.log('Multidemo detected');
    const multidemo = document.createElement("div");
    multidemo.id = "_multidemo";
    for (let i=0;;++i) {
        if (params.get('p'+i)) {
          const name=params.get('p'+i);
          const value=params.get('p'+i+'v');
          const option = document.createElement("input");
          option.type = "radio";
          option.id = "_multidemo_option"+i;
          option.name = "_multidemo_toggle";
          option.checked = value === defaultParams;
          option.onchange = () => {
            params.set('p', value);
            window.location.replace(window.location.pathname + '?' + params.toString());
          };
          multidemo.appendChild(option);
          const label = document.createElement("label");
          label.htmlFor = "_multidemo_option"+i;
          label.innerText = name;
          multidemo.appendChild(label);
        } else {
          break;
        }
    }
    document.body.appendChild(multidemo);
}
"""


def make_multidemo(demo: str):
    """
    Converts single-demo to multi-demo.

    Args:
        demo: The demo to generate the multi-demo from.
    """

    with open(os.path.join(demo, "index.html"), "r") as f:
        html = f.read()
        html = html.replace("</head>", f"<style>{_css}</style></head>")
        html = html.replace("</body>", f"<script>{_js}</script></body>")
    with open(os.path.join(demo, "index.html"), "w") as f:
        f.write(html)
