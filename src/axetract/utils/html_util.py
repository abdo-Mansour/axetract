import requests
import re
from htmlrag import clean_html as clean_html_rag
from html_chunking import get_html_chunks
from bs4 import BeautifulSoup, NavigableString, Tag, Comment
from typing import List, Dict, Optional, Union, Tuple, Set
from difflib import SequenceMatcher

from lxml import etree, html
import copy

class SmartHTMLProcessor:
    def __init__(self):
        self.ATOMIC_TAGS = {'table', 'ul', 'ol', 'dl', 'pre', 'code', 'figure'}
        self.BLOCK_TAGS = {
            'article', 'aside', 'blockquote', 'div', 'fieldset', 'figure', 
            'footer', 'form', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'header', 
            'main', 'nav', 'noscript', 'p', 'section', 'li'
        }
        self.INLINE_TAGS = {'span', 'a', 'b', 'strong', 'i', 'em', 'u', 'sub', 'sup', 'br', 'img', 'q', 'small', 'big', 'cite', 'label'}
        self.ALLOWED_ATTRS = {'href', 'src', 'alt', 'colspan', 'rowspan', 'title', 'target'}

    def _normalize_whitespace(self, text, preserve_newlines=False):
        if not text: return ""
        if preserve_newlines: return text.strip()
        return re.sub(r'\s+', ' ', text).strip()

    def _clean_element(self, element, is_root=False):
        etree.strip_elements(element, 'script', 'style', 'meta', 'noscript', 'link', 'svg', 'iframe')
        etree.strip_tags(element, etree.Comment)
        for node in element.iter():
            keys = list(node.attrib.keys())
            for key in keys:
                if key not in self.ALLOWED_ATTRS:
                    del node.attrib[key]
                elif key == 'href' and node.attrib[key].strip().lower().startswith('javascript:'):
                    del node.attrib[key]
        if is_root: element.tail = None

    def _element_to_string(self, element):
        return html.tostring(element, encoding='unicode', pretty_print=False)

    def _is_layout_table(self, element):
        if element.tag != 'table': return False
        if element.find('.//th') is not None: return False
        if element.find('.//table') is not None: return True
        if len(element.findall('.//tr')) < 2: return True
        return False

    def extract_chunks(self, html_content):
        if not isinstance(html_content, str) or not html_content.strip(): return []
        try:
            root = html.fromstring(html_content)
            tree = root.getroottree()
        except: return []

        chunks = []
        buffer = []
        counter = {'id': 0}

        def flush_buffer(element_xpath):
            if not buffer: return
            raw = "".join(buffer)
            clean = self._normalize_whitespace(raw)
            if clean and (len(clean) > 1 or '<img' in raw):
                chunks.append({
                    'id': counter['id'],
                    'xpath': element_xpath, 
                    'content': clean,
                    'type': 'text'
                })
                counter['id'] += 1
            buffer.clear()

        def traverse(element):
            if element.text: buffer.append(element.text)

            for child in element:
                if not isinstance(child.tag, str): 
                    if child.tail: buffer.append(child.tail)
                    continue

                tag = child.tag
                is_atomic = tag in self.ATOMIC_TAGS
                is_block = tag in self.BLOCK_TAGS
                is_layout = is_atomic and self._is_layout_table(child)

                if is_layout or (tag not in self.BLOCK_TAGS and tag not in self.ATOMIC_TAGS and tag not in self.INLINE_TAGS):
                    flush_buffer(tree.getpath(element))
                    traverse(child)
                    if child.tail: buffer.append(child.tail)
                    continue

                if is_block or is_atomic:
                    flush_buffer(tree.getpath(element))

                    if is_atomic:
                        child_copy = copy.deepcopy(child)
                        self._clean_element(child_copy, is_root=True)
                        clean_html = self._normalize_whitespace(self._element_to_string(child_copy), preserve_newlines=(tag=='pre'))
                        
                        chunks.append({
                            'id': counter['id'],
                            'xpath': tree.getpath(child),
                            'content': clean_html,
                            'type': 'atomic'
                        })
                        counter['id'] += 1
                    else:
                        has_block_children = any((c.tag in self.BLOCK_TAGS or c.tag in self.ATOMIC_TAGS) and not self._is_layout_table(c) for c in child)
                        if has_block_children:
                            traverse(child)
                        else:
                            child_copy = copy.deepcopy(child)
                            self._clean_element(child_copy, is_root=True)
                            if bool(child_copy.text_content().strip()) or child_copy.find('.//img') is not None:
                                clean_html = self._normalize_whitespace(self._element_to_string(child_copy))
                                chunks.append({
                                    'id': counter['id'],
                                    'xpath': tree.getpath(child),
                                    'content': clean_html,
                                    'type': 'text'
                                })
                                counter['id'] += 1
                
                elif tag == 'br':
                    buffer.append(" ")
                elif tag in self.INLINE_TAGS:
                    child_copy = copy.deepcopy(child)
                    self._clean_element(child_copy, is_root=True)
                    buffer.append(self._element_to_string(child_copy))

                if child.tail: buffer.append(child.tail)

            flush_buffer(tree.getpath(element))

        traverse(root)
        return chunks

    def reconstruct_skeleton(self, original_html, selected_chunks):
        if not selected_chunks: return ""
        merged_list = []
        for chunk in selected_chunks:
            # print("CURRENT CHUNK",chunk)
            if len(chunk) == 0:
                continue
            for c in chunk:
                # print("CURRENT C",c)
                # if len(c) == 0:
                #     continue
                # if isinstance(c,pl.Series):
                #     merged_list.append((c[0], c[1]))
                # else:
                #     merged_list.append((c['xpath'], c['content']))
                merged_list.append(c)
        selected_chunks = merged_list

        original_html = clean_html(original_html)
        print("Selected Chunks: ",selected_chunks)
        print("HTML : ",original_html)
        root = html.fromstring(original_html)
        tree = root.getroottree()
        
        selected_xpaths = {c['xpath'] for c in selected_chunks}
        
        # Identify structural ancestors to keep
        kept_nodes = set()
        for chunk in selected_chunks:
            nodes = tree.xpath(chunk['xpath'])
            if not nodes: continue
            node = nodes[0]
            kept_nodes.add(node)
            for ancestor in node.iterancestors():
                kept_nodes.add(ancestor)
                
        def prune(element, parent_is_selected=False):
            element_xpath = tree.getpath(element)
            is_explicitly_selected = element_xpath in selected_xpaths
            
            # Keep text if selected OR if inline inside selected
            should_keep_text = is_explicitly_selected or (parent_is_selected and element.tag in self.INLINE_TAGS)

            # Wipe text if not kept
            if not should_keep_text:
                element.text = None

            to_remove = []
            
            for child in element:
                # 1. Structural Child (Block/Atomic)
                if child in kept_nodes:
                    # Recurse (Reset selection context unless child itself is selected)
                    prune(child, parent_is_selected=False)
                    
                    if not should_keep_text:
                        child.tail = None
                    continue
                
                # 2. Inline Child (Content)
                if should_keep_text and child.tag in self.INLINE_TAGS:
                    # Pass context down!
                    prune(child, parent_is_selected=True)
                    continue
                
                to_remove.append(child)

            for child in to_remove:
                element.remove(child)

        prune(root)
        return html.tostring(root, encoding='unicode', pretty_print=True)

# ==========================================
# TEST RUN
# ==========================================
# if __name__ == "__main__":
#     processor = SmartHTMLProcessor()
    
#     html_doc = """
#     <div>
#     yesssss
#     <p>Price: <p>ds</p> <span> 200 </span></p>
#     noooo
#     <span>$19.99</span>
#     </div>
#     """
    
#     print("--- ORIGINAL HTML ---")
#     print(html_doc.strip())
    
#     chunks = processor.extract_chunks(html_doc)
    
#     print("\n--- EXTRACTED CHUNKS ---")
#     for c in chunks:
#         print(f"ID {c['id']} | XPath: {c['xpath']} | {c['content']}")

#     skeleton = processor.reconstruct_skeleton(html_doc, chunks)

#     print("\n--- RECONSTRUCTED SKELETON ---")
#     print(skeleton)

def merge_html_chunks(chunks: List[str], fallback_content: str) -> str:
        """
        Merge a list of HTML chunk strings into a single HTML document.
        If a chunk has no <body>, append its top-level nodes anyway.
        Returns prettified HTML (we later strip newlines).
        """

        # print("chunks to merge:")
        # print(chunks)
        # print(isinstance(chunks[0],list))
        # print(type(chunks[0]))
        if not isinstance(chunks[0],str):
            merged_list = []
            for chunk in chunks:
                # print("CURRENT CHUNK",chunk)
                if len(chunk) == 0:
                    continue
                for c in chunk:
                    # print("CURRENT C",c)
                    if len(c) == 0:
                        continue
                    merged_list.append((c['xpath'], c['content']))
                    
            if len(merged_list) == 0:
                print("FALLBACK USED")
                final_content = clean_html(fallback_content)
                return final_content
            # print("MERGINGGG")
            # print(merged_list)
            final_html = merge_xpaths_to_html(merged_list)
            final_html = clean_html(final_html)
        else:
            final_html = ""

            for i , ch in enumerate(chunks):
                if ch.strip() == "":
                    continue
                final_html += f"\nchunk number {i}"
                final_html += ch
        
        return final_html

def normalize_html_text(text: str) -> str:
    """
    Normalize text by:
    - Converting weird Unicode whitespace (e.g. \u00a0, \u2009) to normal spaces
    - Collapsing multiple spaces into one
    - Stripping leading/trailing spaces
    Keeps capitalization and punctuation unchanged.
    """
    if not text:
        return ""
    
    # Normalize Unicode form
    normalized = text
    # Replace any Unicode whitespace with a plain space
    normalized = "".join(" " if ch.isspace() else ch for ch in normalized)
    
    # Collapse multiple spaces
    normalized = re.sub(r" +", " ", normalized)
    
    return normalized.strip()

DEFAULT_REMOVE_TAGS = ("script", "style")

def fetch_content( url: str,
                   timeout = 15) -> str:
    """
    Fetch the raw HTML content of a web page.

    Args:
        url (str): The URL of the web page to fetch.
        timeout (int, optional): The maximum time (in seconds) to wait for a response. 
                                Defaults to 15.

    Returns:
        str: The HTML content of the page if the request succeeds.
            If an error occurs, a string prefixed with "[FETCH_ERROR]" and the error message.
    """
    
    headers = {
        "User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    try:
        r = requests.get(url, headers=headers, timeout=timeout)
        r.raise_for_status()
        return r.text
    except Exception as e:
        return f"[FETCH_ERROR] {e}"
        


def clean_html( html_content: str,
                extra_remove_tags = ["header" ,"footer"],
                strip_attrs: bool = True,
                strip_links: bool = True,
                keep_tags: bool = True,
                use_clean_rag: bool = True) -> str:
    """
    Clean raw HTML content by removing unwanted tags, attributes, comments, and optionally links.

    Args:
        html_content (str): The raw HTML content to clean.
        extra_remove_tags (List[str], optional): Additional tags to remove besides the defaults
                                                 (`script`, `style`). Defaults to ["header", "footer"].
        strip_attrs (bool, optional): If True, remove all tag attributes (e.g., class, id). 
                                      Defaults to True.
        strip_links (bool, optional): If True, replace <a> tags with their inner text. Defaults to True.
        keep_tags (bool, optional): If True, return cleaned HTML (with tags preserved). 
                                    If False, return plain text only. Defaults to True.
        use_clean_rag (bool, optional): If True, apply `htmlrag.clean_html` for additional 
                                        normalization. Defaults to True.

    Returns:
        str: The cleaned HTML or plain text, depending on `keep_tags`.
    """
    html_content = custom_clean_html(html_content)
    # soup = BeautifulSoup(html_content or "", "html.parser")

    # remove_tags = set(DEFAULT_REMOVE_TAGS) | set(extra_remove_tags)
    # for tag_name in remove_tags:
    #     for tag in soup.find_all(tag_name):
    #         tag.decompose()

    # for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
    #     comment.extract()

    # if strip_attrs:
    #     for tag in soup.find_all(True):
    #         tag.attrs = {}

    # if strip_links:
    #     for a in soup.find_all('a'):
    #         a.replace_with(a.get_text())

    # for tag in soup.find_all(True):
    #     if not tag.get_text(strip=True):
    #         tag.decompose()

    # if keep_tags:
    #     html_str = str(soup)
    #     html_str = re.sub(r'(?m)^[ \t]*\n', '', html_str)
    #     return html_str.strip()

    # text = soup.get_text(separator='\n', strip=True)
    # lines = [line for line in text.splitlines() if line.strip()]
    # clean_text = '\n'.join(lines)
    # clean_text = re.sub(r'\s+', ' ', clean_text)
    
    clean_text = html_content
    if use_clean_rag:
        clean_text = clean_html_rag(clean_text)

    return clean_text.strip()

def custom_clean_html(html_content: str) -> str:
    """Clean HTML content by removing scripts/styles/comments and invisible elements.
    
    The function removes elements that are likely invisible (inline styles with 
    display:none, visibility:hidden, opacity:0, hidden attribute, aria-hidden).
    It also removes <script>, <style>, <noscript>, <template>, <iframe>, and 
    other non-visible tags, plus all comments.
    
    Args:
        html_content: raw HTML string
    
    Returns:
        cleaned HTML string (serialized back to HTML)
    """
    parser = html.HTMLParser(remove_comments=False)
    doc = html.fromstring(html_content, parser=parser)
    
    # Remove comment nodes
    _remove_comments(doc)
    
    # Remove unwanted tags in one pass
    unwanted_tags = [
        "script", "style", "noscript", "iframe", 
        "object", "embed", "canvas", "svg",
    ]
    _remove_nodes_by_tag(doc, unwanted_tags)
    
    # Regex to detect hidden inline styles
    HIDDEN_REGEX = re.compile(
        r'display\s*:\s*none|visibility\s*:\s*hidden|opacity\s*:\s*0',
        re.IGNORECASE
    )
    
    # Collect hidden elements
    to_remove = []
    for el in doc.iter():
        # Skip non-element nodes (comments, processing instructions, etc.)
        if not isinstance(el.tag, str):
            continue
        
        # Check for 'hidden' attribute
        if el.get('hidden') is not None:
            to_remove.append(el)
            continue
        
        # Check for aria-hidden="true"
        if el.get('aria-hidden') == 'true':
            to_remove.append(el)
            continue
        
        # Check inline style for hiding
        style = el.get('style', '')
        if style and HIDDEN_REGEX.search(style):
            to_remove.append(el)
            continue
        
    
    # Remove hidden elements
    for el in to_remove:
        parent = el.getparent()
        if parent is not None:
            try:
                parent.remove(el)
            except Exception:
                pass
    
    # Remove CSS/JS event attributes from remaining elements
    css_js_attrs = [
        'style', 'onclick', 'onload', 'onmouseover', 'onmouseout', 
        'onchange', 'onsubmit', 'onfocus', 'onblur', 'onkeydown', 
        'onkeyup', 'onkeypress', 'onerror', 'onabort', 'onreset', 
        'onselect', 'onunload', 'onresize', 'onmouseenter', 
        'onmouseleave', 'onwheel', 'oninput', 'ondrag', 'ondrop', 
        'oncontextmenu', 'oncopy', 'oncut', 'onpaste'
    ]
    
    for el in doc.iter():
        if not isinstance(el.tag, str):
            continue
        for attr in css_js_attrs:
            if attr in el.attrib:
                del el.attrib[attr]
    
    # Return normalized HTML string
    return etree.tostring(doc, encoding="unicode", method="html")

def _remove_comments(doc):
    """Remove all HTML comment nodes from the document tree."""
    comments = doc.xpath('//comment()')
    for comment in comments:
        parent = comment.getparent()
        if parent is not None:
            parent.remove(comment)


def _remove_nodes_by_tag(doc, tag_names):
    """Remove all elements with specified tag names.
    
    Args:
        doc: lxml document tree
        tag_names: list of tag names to remove (e.g., ['script', 'style'])
    """
    for tag in tag_names:
        for el in doc.xpath(f'.//{tag}'):
            parent = el.getparent()
            if parent is not None:
                try:
                    parent.remove(el)
                except Exception:
                    pass

def chunk_html_content( html_content: str,
                        max_tokens: int = 500,
                        is_clean: bool = True,
                        attr_cutoff_len: int = 5) -> List[str]:
    """
    Split HTML content into smaller chunks suitable for processing (e.g., with LLMs).

    Args:
        html_content (str): The HTML content to chunk.
        max_tokens (int, optional): Maximum token length per chunk. Defaults to 500.
        is_clean (bool, optional): Whether the input HTML is already cleaned. Defaults to True.
        attr_cutoff_len (int, optional): Maximum length of attributes to retain. Defaults to 5.

    Returns:
        List[str]: A list of HTML/text chunks.
    """
    if not html_content:
        return []
    return get_html_chunks(html=html_content, max_tokens=max_tokens, is_clean_html=is_clean, attr_cutoff_len=attr_cutoff_len)


def normalize_text(s: str) -> str:
    """
    Lowercase, replace punctuation with spaces, and collapse whitespace.
    This makes substring checks more robust (e.g. 6'10" -> '6 10').
    """
    if not s:
        return ""
    # replace non-word chars with spaces, lowercase, collapse spaces
    cleaned = re.sub(r'[^\w\s]', ' ', s).lower()
    return re.sub(r'\s+', ' ', cleaned).strip()

def find_closest_html_node(html_text, search_text):
    """
    Return the chunk (and its xpath/sub_index) that:
      - includes the normalized search_text as substring, and
      - has the highest fuzzy score among all such chunks.

    If no chunk includes search_text, returns {'text': search_text, 'xpath': None, 'sub_index': None, 'score': 0.0, 'found': False}.
    """
    soup = BeautifulSoup(html_text, 'html.parser')
    norm_search = normalize_text(search_text)

    best_containing_score = 0.0
    best_containing_subset = 0
    best_containing_element = None
    best_containing_chunk = None
    best_containing_sub_index = None

    

    if not norm_search:
        # nothing meaningful to search for — return not-found payload
        return {'text': search_text, 'xpath': None, 'sub_index': None, 'score': 0.0, 'found': False}

    # Iterate only once: check containment and compute fuzzy score for those that contain
    for element in soup.find_all(True):
        for idx, chunk in enumerate(get_text_chunks(element)):
            if not chunk or not chunk.strip():
                continue
            intersection_tokens = len(set(norm_search.split()) & set(normalize_text(chunk).split()))
            if (norm_search in normalize_text(chunk)) or intersection_tokens:
                # candidate includes the search text -> compute fuzzy score
                score = SequenceMatcher(None, chunk.strip(), search_text.strip()).ratio()
                # print( score , intersection_tokens , chunk.strip())
                if score >= best_containing_score and intersection_tokens >= best_containing_subset:
                    best_containing_score = score
                    best_containing_subset = intersection_tokens
                    best_containing_element = element
                    best_containing_chunk = chunk.strip()
                    best_containing_sub_index = idx

    if best_containing_element is None:
        # nothing included the search text
        return {'text': search_text, 'xpath': None, 'sub_index': None, 'score': 0.0, 'found': False}

    return {
        'text': best_containing_chunk,
        'xpath': get_xpath(best_containing_element),
        'sub_index': best_containing_sub_index,
        'score': best_containing_score,
        'found': True
    }



def get_text_chunks(element):
    """
    Split by tags (include inner-tag text as separate chunks).
    """
    chunks = []
    buf = []

    for content in element.contents:
        if isinstance(content, NavigableString) and not isinstance(content, Comment):
            buf.append(str(content))
        elif isinstance(content, Tag):
            if buf:
                chunks.append(''.join(buf).strip())
                buf = []
            tag_text = content.get_text(separator=' ', strip=True)
            if tag_text:
                chunks.append(tag_text)
        else:
            if buf:
                chunks.append(''.join(buf).strip())
                buf = []

    if buf:
        chunks.append(''.join(buf).strip())

    return [c for c in chunks if c]


def get_xpath(element):
    components = []
    child = element if element.name else element.parent

    for parent in child.parents:
        siblings = parent.find_all(child.name, recursive=False)
        if len(siblings) == 1:
            components.append(child.name)
        else:
            index = siblings.index(child) + 1
            components.append(f'{child.name}[{index}]')
        child = parent

    components.reverse()
    if components and components[0] == '[document]':
        components.pop(0)

    return '/' + '/'.join(components)




def _normalize_whitespace(s: str) -> str:
    return " ".join(s.split())

# You probably already have _is_visible_element in your code.
# If not, a minimal sensible one:
def _is_visible_element(el) -> bool:
    # skip elements hidden by attributes
    if el.get("hidden") is not None: 
        return False
    if el.get("aria-hidden") == "true":
        return False
    # tag filter — keep consistent with your original
    if el.tag is None:
        return False
    return True

def extract_visible_xpaths_leaves(
    cleaned_html: str,
    min_length: int = 1,
    dedupe_texts: bool = True
) -> List[Tuple[str, str]]:
    """
    Extract visible text preserving exact reading order.
    Interleaves parent text, child elements, and child tails correctly.
    """
    try:
        tree = html.fromstring(cleaned_html)
    except etree.ParserError:
        return []
        
    roottree = tree.getroottree()
    results: List[Tuple[str, str]] = []
    seen_texts: Set[str] = set()
    
    # Tags that definitely don't contain visible text
    skip_tags = {"script", "style", "noscript", "template", "head", "meta", "link"}

    def _process_node(el: html.HtmlElement):
        """
        Recursive function to visit nodes in reading order:
        1. Element Text
        2. Child 1 -> (Recurse) -> Child 1 Tail
        3. Child 2 -> (Recurse) -> Child 2 Tail
        """
        if not isinstance(el.tag, str):
            return

        # 1. Check strict visibility of the tag itself
        if el.tag.lower() in skip_tags:
            return
        
        # Optional: Add your _is_visible_element(el) check here if you have it
        # if not _is_visible_element(el): return

        xpath = roottree.getpath(el)

        # --- A. Handle Text inside this element (before first child) ---
        if el.text:
            text = _normalize_whitespace(el.text)
            if len(text) >= min_length:
                if not (dedupe_texts and text in seen_texts):
                    results.append((xpath, text))
                    seen_texts.add(text)

        # --- B. Handle Children and their Tails ---
        for child in el:
            if not isinstance(child.tag, str): 
                continue
                
            # 1. Recurse into child (this captures the child's internal text)
            _process_node(child)
            
            # 2. Handle the Child's Tail 
            # (Visually, this appears AFTER the child, but belongs to the PARENT's XPath)
            if child.tail:
                tail_text = _normalize_whitespace(child.tail)
                if len(tail_text) >= min_length:
                    if not (dedupe_texts and tail_text in seen_texts):
                        # Note: The text physically follows 'child', but structurally belongs to 'el' (parent)
                        results.append((xpath, tail_text))
                        seen_texts.add(tail_text)

    # Start recursion
    _process_node(tree)
    
    return results


# import re
# from typing import List, Tuple
# from lxml import etree, html

_INDEXED_STEP_RE = re.compile(r"^([A-Za-z0-9_\-\.:]+)(?:\[(\d+)\])?$")

def merge_xpaths_to_html(
    xpath_text_list: List[Tuple[str, str]],
    root_tag: str = "html",
    pretty: bool = True
) -> str:
    """
    Merge a list of (absolute_xpath, content) into one HTML document.
    """
    
    # --- 1. Setup Root ---
    root = etree.Element(root_tag)
    if root_tag.lower() == 'html':
        etree.SubElement(root, "head")
        etree.SubElement(root, "body")

    def parse_step(step: str):
        """Parse 'div[2]' -> ('div', 2)."""
        m = _INDEXED_STEP_RE.match(step)
        if not m:
            return step, 1
        tag = m.group(1)
        idx = int(m.group(2)) if m.group(2) else 1
        return tag, idx

    def get_nth_child(parent: etree._Element, tag: str, index: int) -> etree._Element:
        """
        Return (and create if missing) the Nth child of a specific tag.
        """
        # Find existing children with this tag
        matches = [c for c in parent if isinstance(c.tag, str) and c.tag == tag]
        
        if len(matches) >= index:
            return matches[index - 1]
        
        # If the element doesn't exist, create intermediates
        while len(matches) < index:
            try:
                new = etree.Element(tag)
            except ValueError:
                # Fallback for invalid tags (e.g., 'fb:login-button' -> 'fb_login-button')
                safe_tag = tag.replace(":", "_").replace(".", "_")
                # If it's still invalid, fallback to generic div to prevent crash
                try:
                    new = etree.Element(safe_tag)
                except ValueError:
                    new = etree.Element("div", attrib={"original_tag": tag})
            
            parent.append(new)
            matches.append(new)
        return matches[index - 1]

    # --- 2. Process Items in Input Order ---
    for raw_xpath, content in xpath_text_list:
        if not raw_xpath or not raw_xpath.startswith("/"):
            continue

        steps = [s for s in raw_xpath.split("/") if s]
        
        cur = root
        step_index = 0
        
        # Skip root tag in path if it matches our root element
        if steps and steps[0].lower() == root_tag.lower():
            step_index = 1 

        # Traverse or Build the path to the target element
        for step in steps[step_index:]:
            tag, idx = parse_step(step)
            
            # SANITIZE: lxml hates colons in tag names without namespace maps
            if ":" in tag:
                tag = tag.replace(":", "_")
            
            cur = get_nth_child(cur, tag, idx)

        # --- 3. Intelligent Content Insertion ---
        if content is None:
            continue
        piece = content.strip()
        if not piece:
            continue

        try:
            frags = html.fragments_fromstring(piece)
        except (ValueError, etree.ParserError):
            frags = [piece]

        for frag in frags:
            if isinstance(frag, str):
                if len(cur) > 0:
                    last_child = cur[-1]
                    last_child.tail = (last_child.tail or "") + (" " if last_child.tail else "") + frag
                else:
                    cur.text = (cur.text or "") + (" " if cur.text else "") + frag
            else:
                cur.append(frag)

    return etree.tostring(root, method="html", encoding="unicode", pretty_print=pretty)