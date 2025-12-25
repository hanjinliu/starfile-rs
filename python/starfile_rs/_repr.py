from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from starfile_rs.core import StarDict


def html_block(star: StarDict, max_blocks: int = 20000) -> str:
    """Generate HTML representation for a single data block."""
    html_parts = []
    for i_block, (name, block) in enumerate(star.items()):
        block_html = block._rust_obj.to_html(cell_style="padding: 4px;", max_lines=60)
        html_parts.append(_SECTION_FORMAT.format(name=name, block_html=block_html))
        if i_block + 1 >= max_blocks:
            html_parts.append(
                f"<p>... (truncated, {len(star) - max_blocks} more blocks)</p>"
            )
            break
    return "".join(html_parts)


_SECTION_FORMAT = """
<details open>
    <summary style="background-color: #e8f4f8; padding: 3px; cursor: pointer; border-radius: 4px;"><b>{name}</b></summary>
    <div style="font-family: monospace; background-color: #f9f9f9; padding: 4px; border-radius: 4px; animation: slideDown 0.3s ease-out;">
        <style>
            table {{ border-collapse: collapse; }}
            table, th, td {{ border: 1px solid #ddd; }}
        </style>
        {block_html}
    </div>
</details>
"""
