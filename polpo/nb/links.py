from concurrent.futures import ThreadPoolExecutor

import requests


def _thread_map(func, inputs, workers=16):
    n = len(inputs)
    if n == 0:
        return []
    if n == 1:
        return [func(inputs[0])]

    workers = min(workers, n)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(func, inputs))


def is_link_ok(url):
    return requests.get(
        url,
        allow_redirects=True,
        headers={"User-Agent": "link-checker"},
    ).ok


def are_links_ok(urls, workers=16):
    return _thread_map(is_link_ok, urls, workers=workers)
