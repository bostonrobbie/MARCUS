
from typing import Dict, Any
import jinja2

class PineScriptHelper:
    """
    Helper to inject Python parameters into Pine Script Templates.
    """
    
    @staticmethod
    def fill_template(template_str: str, params: Dict[str, Any]) -> str:
        # Convert params to Pine-friendly format
        pine_params = {}
        for k, v in params.items():
            if isinstance(v, bool):
                pine_params[k] = "true" if v else "false"
            elif isinstance(v, str):
                pine_params[k] = f'"{v}"'
            else:
                pine_params[k] = v
                
        # Use Jinja2 for replacement
        template = jinja2.Template(template_str)
        return template.render(**pine_params)
