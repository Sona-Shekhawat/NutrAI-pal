import scrapy
import json

with open("links.json","r")as f:
         links=json.load(f)

class HealthyRecipeSpider(scrapy.Spider):
    name = 'healthy_recipe_spider'
    start_urls=links

  

    def parse(self, response):
        def safe_extract_text(selector):
            return selector.get().strip() if selector else None

        # Extract nutrients as a dictionary
        nutrients = {}
        for li in response.css('div.nutritional li'):
            name = li.css('span.big-nut::text').get()
            full_text = li.css('::text').getall()
            if name:
                value = ''.join(full_text).replace(name, '').strip()
                nutrients[name] = value

        # Extract health tags
        tags = []
        for a in response.css('div.inner.health-info a.recipe_circle'):
            title = a.attrib.get('title', '')
            if "Click for more" in title:
                tag = title.replace("Click for more", "").replace("recipes", "").strip()
                tags.append(tag)

        yield {
            'name': safe_extract_text(response.css('h1.title.fn::text')),
            'ingredients': response.css('div#ingredients-list li.ingredient::text').getall(),
            'instructions': response.css('div#fld_instructions_and_steps p::text').getall(),
            'nutrition': nutrients,
            'time': safe_extract_text(response.css('div.cooking_time_text::text')).replace("Time to make:", "").strip(),
            'serving_size': (safe_extract_text(response.css('div.RECIPE_META_servings::text')) or "").replace("Serves:", "").strip() or None,
            'tags': tags
        }
#  scrapy runspider healthy_recipes_spider.py -o recipes.json