"""
Unified class mapping for cross-platform GUI element detection.

Maps Rico (mobile) and WebUI (web) element classes to a common taxonomy
of 12 classes that exist across both platforms.
"""

# 12 unified classes for cross-platform GUI detection
UNIFIED_CLASSES = [
    "Button",
    "Text",
    "Image",
    "Icon",
    "Input",
    "Link",
    "Checkbox",
    "Toggle",
    "Toolbar",
    "Navigation",
    "Modal",
    "Tab",
]

# Rico componentLabel → unified class
# Source: "Learning Design Semantics for Mobile Apps" (Liu et al., 2018)
# 25 component categories from Rico semantic annotations
RICO_TO_UNIFIED = {
    "Text Button":       "Button",
    "Text":              "Text",
    "Image":             "Image",
    "Icon":              "Icon",
    "Input":             "Input",
    "EditText":          "Input",
    "Checkbox":          "Checkbox",
    "CheckedTextView":   "Checkbox",
    "Radio Button":      "Checkbox",
    "Switch":            "Toggle",
    "On/Off Switch":     "Toggle",
    "Toolbar":           "Toolbar",
    "Bottom Navigation": "Navigation",
    "Bottom-Navigation": "Navigation",
    "Drawer":            "Navigation",
    "Modal":             "Modal",
    "Multi-Tab":         "Tab",
    "Multi-tab":         "Tab",
    "Tab":               "Tab",
    "Slider":            "Input",
    "Spinner":           "Input",
    "Date Picker":       "Input",
    "Number Stepper":    "Input",
    "Video":             "Image",
    "List Item":         None,   # Skip - too generic
    "Card":              None,   # Skip - container
    "Web View":          None,   # Skip - embedded content
    "Map View":          None,   # Skip - embedded content
    "Background Image":  None,   # Skip - not interactive
    "Page Indicator":    None,   # Skip - decorative
    "PageIndicator":     None,
    "Pager Indicator":   None,
    "Upper Task Bar":    None,   # Skip - system UI
    "UpperTaskBar":      None,
    "Button Bar":        "Toolbar",
    "Advertisement":     None,   # Skip - external content
    "Remember":          None,   # Skip - VINS specific
}

# WebUI ARIA roles → unified class
# Source: W3C Accessibility Tree roles extracted by WebUI crawler
WEBUI_TO_UNIFIED = {
    "button":         "Button",
    "link":           "Link",
    "heading":        "Text",
    "img":            "Image",
    "image":          "Image",
    "textbox":        "Input",
    "combobox":       "Input",
    "searchbox":      "Input",
    "spinbutton":     "Input",
    "slider":         "Input",
    "checkbox":       "Checkbox",
    "radio":          "Checkbox",
    "switch":         "Toggle",
    "toolbar":        "Toolbar",
    "navigation":     "Navigation",
    "menubar":        "Navigation",
    "menu":           "Navigation",
    "dialog":         "Modal",
    "alertdialog":    "Modal",
    "tab":            "Tab",
    "tablist":        None,   # Skip - container for tabs
    "StaticText":     "Text",
    "paragraph":      "Text",
    "strong":         "Text",
    "emphasis":       "Text",
    "listitem":       None,   # Skip - container
    "list":           None,
    "article":        None,
    "section":        None,
    "generic":        None,
    "none":           None,
    "contentinfo":    None,
    "complementary":  None,
    "main":           None,
    "banner":         None,
    "form":           None,
    "region":         None,
    "group":          None,
    "separator":      None,
    "LineBreak":      None,
    "RootWebArea":    None,
    "ListMarker":     None,
    "Section":        None,
    "LayoutTableCell": None,
    "gridcell":       None,
    "row":            None,
    "time":           None,
    "figure":         None,
    "cell":           None,
    "table":          None,
    "columnheader":   None,
    "rowgroup":       None,
    "rowheader":      None,
    "definition":     None,
    "term":           None,
    "code":           None,
    "blockquote":     None,
    "IframePresentational": None,
    "Iframe":         None,
}


def get_class_id(class_name):
    """Get the integer ID for a unified class name. Returns None if not found."""
    try:
        return UNIFIED_CLASSES.index(class_name)
    except ValueError:
        return None


def map_rico_label(component_label):
    """Map a Rico componentLabel to (unified_class_name, class_id) or (None, None)."""
    unified = RICO_TO_UNIFIED.get(component_label)
    if unified is None:
        return None, None
    return unified, get_class_id(unified)


def map_webui_label(aria_role):
    """Map a WebUI ARIA role to (unified_class_name, class_id) or (None, None)."""
    unified = WEBUI_TO_UNIFIED.get(aria_role)
    if unified is None:
        return None, None
    return unified, get_class_id(unified)
