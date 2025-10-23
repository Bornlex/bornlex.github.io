+++
title = 'Scraping - Refreshing Frequency'
date = 2025-10-23T23:41:36+02:00
draft = false
+++

Scraping is a topic I like. Gathering data and making sure they are fresh is an interesting problem.

Among the many things that you have to do when scraping, making sure your data is fresh is an important one. Let’s give a few tips to our fellow scraper.

## Problem definition

We are building a intelligent (hopefully) system that optimizes when we should visit an URL. We don’t want to spend too much resource on refreshing too often, and we want our data to be as fresh as possible.

In its most basic form, we are looking for such a function :

$$
f: (u, h) \to d, (u, h, d) \in (\mathbb{U}, \mathbb{H}, \mathbb{N})
$$

With :

- $\mathbb{U}$ : the space of all URLs
- $\mathbb{H}$ : the space of all HTML pages
- $\mathbb{N}$ : the space of natural integers, in our case it is the delay before next visit

Those are the basic inputs and outputs of the function. Of course, nothing prevents us from adding more.

---

Looking at only the URL and content of the page might not be enough. When making a request to a website, we receive more than the content of the page.

Here are a few things that our system can leverage :

- HTTP headers
- Website sitemap
- RSS feed
- URL
- Page content (HTML)
    - Head/Metadata
    - Body

---

Our system should work on both new URLs and known URLs (URLs that we already visited in the past).

Throughout this report, we will take two domains and specific URLs inside those domains as example :

- https://parissecret.com/que-faire-a-paris-ce-week-end/
- https://www.lebonbon.fr/paris/sorties/

## Data available

### HTTP Headers

Many headers might be interesting for our system :

- Cache-Control, and especially the following directives :
    - max-age → maximum number of seconds the content is considered fresh
    - s-maxage → similar to max-age above
    - no-cache → a good indicator that the content is dynamic
    - no-store → content is highly dynamic, so the browser/user should not cache this page
- Expires
    - Roughly the same as Cache-Control, but it gives an absolute timestamp
- Age
    - The time the content has been in cache for
    - This can be used along with Cache-Control (like Cache-Control is 1h and Age is 30min, then we can refresh in 1h - 30min = 30min)
- ETag
    - Provides a hash of the content of the page, which makes it easy to check whether a page has changed since last time
- Last-Modified
    - Date and time when the resource was last modified, can be used to check whether the resource is the same as last time we checked

HTTP headers and their meanings can be found at the following URL : https://developer.mozilla.org/en-US/docs/Web/HTTP/Reference/Headers

### Sitemap

Sitemaps are goldmines. They can be found either in the robots.txt file (best solution) or directly at the https://domain.com/sitemap.xml (but this can vary).

Considering the website we monitor, the following directives can be found at the very end of the robots.txt :

```html
#Sitemaps
Sitemap: https://parissecret.com/sitemap_index.xml
```

Then request this page, we can see different sitemaps :

| Sitemap | Last Modified |
| --- | --- |
| https://parissecret.com/page-sitemap.xml | 2024-11-14 12:16 +00:00 |
| https://parissecret.com/category-sitemap.xml | 2025-10-23 15:37 +00:00 |
| https://parissecret.com/author-sitemap.xml | 2025-09-01 11:46 +00:00 |
| https://parissecret.com/posts_v2-sitemap.xml | 2019-09-24 16:24 +00:00 |
| https://parissecret.com/posts_v2-sitemap2.xml | 2020-10-02 09:18 +00:00 |
| https://parissecret.com/posts_v2-sitemap3.xml | 2021-06-01 11:11 +00:00 |
| https://parissecret.com/posts_v2-sitemap4.xml | 2022-02-09 17:56 +00:00 |
| https://parissecret.com/posts_v2-sitemap5.xml | 2022-12-20 14:20 +00:00 |
| https://parissecret.com/posts_v2-sitemap6.xml | 2023-11-14 11:16 +00:00 |
| https://parissecret.com/posts_v2-sitemap7.xml | 2024-09-24 12:23 +00:00 |
| https://parissecret.com/posts_v2-sitemap8.xml | 2025-05-20 09:24 +00:00 |
| https://parissecret.com/posts_v2-sitemap9.xml | 2025-10-23 15:37 +00:00 |
| https://parissecret.com/posts-en-sitemap.xml | 2025-02-10 15:44 +00:00 |
| https://parissecret.com/posts-en-sitemap2.xml | 2025-10-08 15:50 +00:00 |
| https://parissecret.com/posts-en-sitemap3.xml | 2025-10-23 15:37 +00:00 |
| https://parissecret.com/posts-it-sitemap.xml | 2025-04-29 10:36 +00:00 |
| https://parissecret.com/posts-it-sitemap2.xml | 2025-10-23 15:37 +00:00 |
| https://parissecret.com/posts-pt-sitemap.xml | 2025-05-19 07:19 +00:00 |
| https://parissecret.com/posts-pt-sitemap2.xml | 2025-10-23 15:37 +00:00 |
| https://parissecret.com/posts-nl-sitemap.xml | 2025-05-19 09:44 +00:00 |
| https://parissecret.com/posts-nl-sitemap2.xml | 2025-10-23 15:37 +00:00 |
| https://parissecret.com/posts-es-sitemap.xml | 2025-02-27 15:57 +00:00 |
| https://parissecret.com/posts-es-sitemap2.xml | 2025-10-23 14:40 +00:00 |
| https://parissecret.com/posts-es-sitemap3.xml | 2025-10-23 15:37 +00:00 |
| https://parissecret.com/posts-de-sitemap.xml | 2025-05-05 15:59 +00:00 |
| https://parissecret.com/posts-de-sitemap2.xml | 2025-10-23 15:37 +00:00 |

One can notice that some sitemaps seem to have not been used for a long time, but requesting the sitemaps that we have been refreshed since last crawl, we can for example use this one : https://parissecret.com/posts_v2-sitemap9.xml and see the following content at the very end of the table :

| https://parissecret.com/que-faire-a-paris-ce-week-end/ | 0 | 2025-10-23 15:37 +00:00 |
| --- | --- | --- |

which is our page. This is particularly interesting because we can directly see the last time it has been modified, allowing us to know whether we need to refresh the page or not.

---

Sometimes, sitemaps directly give frequencies at which articles or sub content are refreshed. For example, the sitemap taken from [https://www.lebonbon.fr](https://www.lebonbon.fr/) domain :

```html
<url>
	<loc>https://www.lebonbon.fr/paris/actu/</loc>
	<lastmod>2025-10-23</lastmod>
	<changefreq>daily</changefreq>
	<priority>0.5</priority>
</url>
<url>
	<loc>https://www.lebonbon.fr/paris/sorties/</loc>
	<lastmod>2025-10-22</lastmod>
	<changefreq>daily</changefreq>
	<priority>0.5</priority>
</url>
```

<aside>
💡

Another good thing with sitemaps is that they offer an overview of the whole website. So instead of requesting each page we monitor and checking the headers, URL, content… we can parse the whole sitemap and have a list of pages that need to be refreshed.

</aside>

### RSS Feed

RSS feeds are interesting because they are well formatted, easily available information pushed by the websites themselves. Al websites do not provide RSS feeds though.

There are multiple ways of finding whether a website has a RSS feed :

- Checking routes
    - /rss
    - /feed
    - /rss.xml
- Looking inside the HTML <head> tag

```html
<rss xmlns:atom="http://www.w3.org/2005/Atom" xmlns:dc="http://purl.org/dc/elements/1.1/" version="2.0">
	<channel>
		<title>L'Équipe - L'actualité du sport en continu.</title>
		<link>https://www.lequipe.fr/</link>
		<description>Le sport en direct sur L'Équipe. Les informations, résultats et classements de tous les sports. Directs commentés, images et vidéos à regarder et à partager</description>
		<language>fr-fr</language>
		<copyright>Copyright L'Equipe.fr</copyright>
		<pubDate>Thu, 23 Oct 2025 13:23:25 +0200</pubDate>
		<item>
			<title>Le débrief du premier match de la saison de Wembanyama en NBA : « Une façon d'annoncer qu'il va tout écraser »</title>
			<link>https://www.lequipe.fr/Basket/Actualites/Le-debrief-du-premier-match-de-la-saison-de-wembanyama-en-nba-une-facon-d-annoncer-qu-il-va-tout-ecraser/1603995#at_medium=RSS_feeds</link>
			<description>
				<![CDATA[ <img src="https://medias.lequipe.fr/img-photo-png/-/1500000000833469/0-828-552-75/0cb67.png" alt="Logo L'ÉQUIPE">Ultra-dominateur face aux Dallas Mavericks pour son retour en NBA après huit mois d'absence, Victor Wembanyama a affiché dès le premier match de la saison des San Antonio Spurs les progrès effectués dans l'intervalle, avec une palette offensive plus complète que jamais. ]]>
			</description>
			<dc:creator>Amaury Perdriau</dc:creator>
			<category>Basket, NBA, Spurs</category>
			<enclosure url="https://medias.lequipe.fr/img-photo-png/-/1500000000833469/0-828-552-75/0cb67.png" length="50000" type="image/jpeg"/>
			<guid>https://www.lequipe.fr/Basket/Actualites/Le-debrief-du-premier-match-de-la-saison-de-wembanyama-en-nba-une-facon-d-annoncer-qu-il-va-tout-ecraser/1603995</guid>
			<pubDate>Thu, 23 Oct 2025 13:21:49 +0200</pubDate>
		</item>
		...
</rss>
```

It is sometimes also possible to transform a page that aggregates posts into a RSS feed using rss.app. For example, using the following page from Le Bonbon : https://www.lebonbon.fr/paris/sorties/ we get an URL pointing to a JSON generated page that looks like :

```json
{
  "version": "https://jsonfeed.org/version/1.1",
  "title": "Précurseur de tendances",
  "home_page_url": "https://www.lebonbon.fr/paris/sorties/",
  "feed_url": "https://rss.app/feeds/v1.1/cCA4EDJMK6KOk6pl.json",
  "favicon": "https://www.lebonbon.fr/favicon/lebonbon/apple-touch-icon.png",
  "language": "fr",
  "description": "Découvre les événements à ne pas manquer près de chez toi : concerts, expositions, festivals, spectacles, soirées",
  "items": [
    {
      "id": "884aac3ebda0c73be65a0e8a778b870b",
      "url": "https://www.lebonbon.fr/paris/bons-plans/que-faire-paris-week-end-24-26-octobre-2025/",
      "title": "Que faire à Paris ce week-end ? (24-26 octobre)",
      "content_text": "De l’art, de la fête, des spectacles et de la food… Les Bonbons, ce week-end sera pluvieux mais généreux – d’autant plus qu’on aura une heure en plus !",
      "content_html": "\u003Cdiv\u003E\u003Cimg src=\"https://uploads.lebonbon.fr/source/2025/october/2083397/jagodabartus-lebonbon_2_1200.jpg\" style=\"width: 100%;\" /\u003E\u003Cdiv\u003EDe l’art, de la fête, des spectacles et de la food… Les Bonbons, ce week-end sera pluvieux mais généreux – d’autant plus qu’on aura une heure en plus !\u003C/div\u003E\u003C/div\u003E",
      "image": "https://uploads.lebonbon.fr/source/2025/october/2083397/jagodabartus-lebonbon_2_1200.jpg",
      "date_published": "2025-10-23T17:00:00.000Z",
      "authors": [
        {
          "name": "Maria Sumalla"
        }
      ],
      "attachments": [
        {
          "url": "https://uploads.lebonbon.fr/source/2025/october/2083397/jagodabartus-lebonbon_2_1200.jpg"
        }
      ]
    },
    {
      "id": "f90e21f1db9a67f263db44ea390e28d3",
      "url": "https://www.lebonbon.fr/paris/loisirs/cette-experience-incroyable-fait-vivre-batailles-napoleon-visiter-france-1800-a-paris/",
      "title": "Cette expérience incroyable fait vivre les batailles de Napoléon et visiter le Paris de l'époque",
      "content_text": "L’empereur fait son grand retour à Paris ! L’expérience immersive Napoléon, l’Épopée Immersive s’installe dans le cadre monumental de la Poste du Louvre, au cœur du 1er arrondissement.",
      "content_html": "\u003Cdiv\u003E\u003Cimg src=\"https://uploads.lebonbon.fr/source/2025/october/2083339/realite-virtuelle-napoleon_2_1200.jpg\" style=\"width: 100%;\" /\u003E\u003Cdiv\u003EL’empereur fait son grand retour à Paris ! L’expérience immersive Napoléon, l’Épopée Immersive s’installe dans le cadre monumental de la Poste du Louvre, au cœur du 1er arrondissement.\u003C/div\u003E\u003C/div\u003E",
      "image": "https://uploads.lebonbon.fr/source/2025/october/2083339/realite-virtuelle-napoleon_2_1200.jpg",
      "date_published": "2025-10-23T16:00:00.000Z",
      "authors": [
        {
          "name": "La Rédac'"
        }
      ],
      "attachments": [
        {
          "url": "https://uploads.lebonbon.fr/source/2025/october/2083339/realite-virtuelle-napoleon_2_1200.jpg"
        }
      ]
    },
```

which is very easy to read.

### Page Metadata

In case previous methods did not work, game is not over yet. The page contains valuable information. Before digging into what we can do with the content itself, let’s have a look at the meta tags contained in the head section of the HTML :

```html
<meta property="article:publisher" content="https://www.facebook.com/paris.secret.sn/" />
<meta property="article:published_time" content="2022-10-13T09:24:19+00:00" />
<meta property="article:modified_time" content="2025-10-20T08:02:28+00:00" />
```

All pages are not equal and some websites might not embed such information, but when they do, it allows us to check easily (before using any LLM) whether the page has been refreshed since last time.

### URL

URLs might contain valuable information that will allow us to guess the nature of the content. For example, we are interested in knowing when to refresh the following page :

```python
"https://parissecret.com/que-faire-a-paris-ce-week-end/"
```

The URL itself contains the word “week-end”. Of course, nothing prevents the authors from adding articles during the week-end, on Mondays… But knowing that the page is going to be talking about what to do during the week-end, we might infer that the content can be refreshed once a day. It is not a news website that required us to monitor it every hour.

This might require the help of an LLM, slowing a bit the process.

### Page Content

Finally, the content of the page. Having to understand the page itself is kind of the worst case scenario. We will definitely have to use an LLM (which is relatively slow) and because LLMs struggle with reading raw HTML (for instance the Huggingface home page is 160k tokens once tokenized, which is more than most models context size), we will have to preprocess the HTML into something that is readable.

But before digging into HTML transformation, let’s look at what we can expect from the content of the page. The parissecret page contains the following meta information about the article :

```html
<a href="/author/lauracoll/" class="notranslate"> Laura Coll</a> - Rédactrice en chef<span class="single__author__bull-divider">&bull;</span>
<time class="single__author__date" datetime="2025-10-20T08:02:28+00:00">octobre 20, 2025</time>
```

The refresh date is indeed present in the page. This is something we can use.

## System

### Intermediary Representation

To transform HTML into something LLMs can ingest, an easy choice is markdown. Markdown is concise and LLMs are trained to see it. The disadvantage is that it flattens the file. The rich structure of the HTML is lost, and in some cases this might create difficulties on its own.

There are several ways of getting Markdown from HTML. It is not too difficult to write our own serialization system, allowing us to remove parts of the HTML that encodes no information such as images, scripts…

A custom override of the LXML parser might look like :

```python
from lxml import etree, html
from lxml.html import HtmlElement
import re

from custom_tags import Transparent, ToRemove, tags_to_remove

class ElementClassLookup(etree.CustomElementClassLookup):
    def __init__(self):
        super().__init__()
        self.tag_to_class_map = load_classes()

    def lookup(self, node_type, doc, namespace, tag_name):
        """
        Either a tag is explicitly handled or it is marked for removal.
        """
        if node_type != 'element':
            return None

        if tag_name in tags_to_remove:
            return ToRemove

        return self.tag_to_class_map.get(tag_name, Transparent)

class Parser(html.HTMLParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_element_class_lookup(ElementClassLookup())
```

And a tag might look like :

```python
from custom_tags import Abstract

class A(Abstract):
    tag = "a"

    def custom_serialize(self, level: int = 0, filtered_tag: list[str] = None, *args, **kwargs):
        """
        <a> has the following attributes:
        - class
        - href
        """
        if self.remove_tag(filtered_tag):
            return ""

        link = self.attrib.get("href", "")
        inner_content = self.serialize_content(with_children=True)

        return f"[{inner_content}]({link})\n"

    def serialize_content(self, with_children: bool = True, filtered_tag: str = None) -> str:
        if self.remove_tag(filtered_tag):
            return ""

        content = ''
        if self.text:
            content = self.text.strip()

        for child in self:
            if with_children:
                content += child.serialize_content(filtered_tag=filtered_tag, with_children=True).strip()

            if child.tail:
                content += self.get_tail().strip()

        return content.strip()
```

The methods serialize_content, get_tail define how we want to serialize the content of the nodes.

### LLM Tool Calling

Once the HTML is transformed into Markdown, the data extraction problem can be reframed as a tool calling problem. We can give a concrete usage of a tool call for this specific case.

Considering the schema contained in the **kwargs is formatted as follows :

```python
{
    'annotations': None,
    'description': 'Extract data from the current page based on the provided schema.',
    'inputSchema': {
        'additionalProperties': False,
        'properties': {
            'date': {
                'type': 'string',
                'description': 'The date at which the page has been last modified.'
            },
            'time': {
                'type': 'string',
                'description': 'The time at which the page has been last modified.',
            }
        },
        'type': 'object'
    },
    'name': "extract_data"
}
```

This code might be used as a first draft/pseudo logic :

```python
class ExtractionProperty(BaseModel):
    type: str
    description: Optional[str] = None

class ExtractionSchema(RootModel[Dict[str, ExtractionProperty]]):
    pass

class ExtractionDataTool():
    """
    This tool is very specific. The schema of its arguments is dynamic and needs to
    be inferred before calling the to_dict() method.
    """
    def __init__(self, *args, **kwargs):
        pass

    def execute(self, schema: dict, markdown_content: str) -> ToolCallResult:
        schema_model = ExtractionSchema.model_validate(schema)

        response = self.extract(
            markdown_content,
            schema_model,
            kwargs['to_extract'],
        )

        if response:
            return ToolCallResult(
                success=True,
                result=f"The extracted data is: {response}."
            )

        return ToolCallResult(
            success=False,
            result="No data extracted. Please check the schema and the content of the page."
        )

    def extract(
            self,
            markdown_content: str,
            schema: ExtractionSchema,
            to_extract: str
    ) -> str:
        formatted_tool_call = {
            messages=self.build_messages(
                markdown_content,
                to_extract
            ),
            tools=[{
                name='extract_data',
                description='Extract data from the current page based on the provided schema.',
                input_schema=FunctionParameters(
                    properties=schema.model_dump(),
                    type='object',
                    additionalProperties=False
                )
            }],
        }

        messages = self.llm_provider.completion(formatted_tool_call)
        results = extract_results_from_messages(messages)

        return results
```

### Frequency Update Formula

As a conceptual exercise, I propose the following formula to update the frequency of scraping. Considering that we have stored every timestamp at which we refreshed the URL and whether the URL changed or not at that timestamp :

$$
x = \left\{ (t_i, x_i) \right\}_{i \lt en}, t_i \in \mathbb{N}, x_i \in \left\{ 0, 1 \right\}
$$

The $x_i$ value is either 0 if the page did not change when we refreshed it, 1 otherwise. We are looking for a refresh interval $f_n$ .

Let’s define the following additional variables :

- $f_n$ the current refresh interval
- $r_n$ the current change rate (the proportion of times we check the page and it changed, for example $r_n = 0.5$ means that 50% of the times we checked the page the content changed since previous time)
- $r_{t}$ the target change rate
    - A target change rate that is too low means that we check the page too often
    - A target change rate that is too high that maybe the page is refreshed with a frequency lower than our scraping frequency, and we save some resources by scraping this page less often
    - We can set the initial target change rate at 0.5
- $\alpha$ the smoothing factor, weights how much the near history matters more than the old history
- $k$ defines how much we change the interval at every update, could start with 0.2

An algorithm might work as follows :

1. We first update the change rate using the smoothing factor :
    
    $r_n = (\alpha x_n) + ((1 - \alpha) r_{n-1})$
    
2. Then we refresh the interval :
    
    $f_n = f_{n-1} (1 - k  (r_n - r_t))$
    

---

The closer $r_n$ is to $r_t$, the lower the difference, and thus the lower is the ajustement we make to the frequency. It is obvious to see that :

$$
\lim_{r_n \to r_t} \Delta f_n = 0
$$

Which is the property we want.
