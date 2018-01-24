#!/usr/bin/env python
# -*- coding: utf-8 -*- #
from __future__ import unicode_literals

AUTHOR = 'Seng Chu'
SITENAME = 'PERCETIVE STUDENT'
#Perceptive student, student perception, perceptivelearner, learningmentaliy
SITEURL = ''
PATH = 'content'
TIMEZONE = 'America/Los_Angeles'
DEFAULT_LANG = 'en'

#Bootstrap specific settings
THEME = "~pelican-themes/pelican-bootstrap3"
JINJA_ENVIRONMENT = {'extensions': ['jinja2.ext.i18n']}
BOOSTRAP_FLUID = True
BOOTSTRAP_THEME = 'cosmo'
PYGMENTS_STYLE = 'default'
#Custom CSS in templates/ipynb.css, injected into base.html

#Social Settings
SOCIAL = (('twitter', ''),
          ('linkedin', ''),
          ('github', ''))

#Bootstrap article settings
SHOW_ARTICLE_CATEGORY = True
		  
#Bootstrap sidebar tags settings		  
DISPLAY_TAGS_ON_SIDEBAR = True
DISPLAY_TAGS_INLINE = False
TAG_CLOUD_MAX_ITEMS = 25
TAG_CLOUD_STEPS = 2
TAG_CLOUD_SORTING = 'random'
TAG_CLOUD_BADGE = True
TAGS_URL = ''

#Bootstrap sidebar other settings
DISPLAY_ARCHIVE_ON_SIDEBAR = True
DISPLAY_RECENT_POSTS_ON_SIDEBAR = True
MONTH_ARCHIVE_SAVE_AS = '{date:%Y}/{date:%m}/index.html'
RECENT_POST_COUNT = 4
DIRECT_TEMPLATES = ['search']
#Edited sidebar title icons, see templates/includes/sidebar

#Bootstrap banner and brand settings
BANNER = 'images/tree.png'
BANNER_ALL_PAGES = True
BANNER_SUBTITLE = 'Engineering the way we see the world'
HIDE_SITENAME = True

#Bootstrap navbar options
DISPLAY_PAGES_ON_MENU = True
DISPLAY_CATEGORIES_ON_MENU = True

# Feed generation is usually not desired when developing
FEED_ALL_ATOM = None
CATEGORY_FEED_ATOM = None
TRANSLATION_FEED_ATOM = None
AUTHOR_FEED_ATOM = None
AUTHOR_FEED_RSS = None


# Blogroll
DEFAULT_PAGINATION = 100


# Uncomment following line if you want document-relative URLs when developing
#RELATIVE_URLS = True

MARKUP = ('md', 'ipynb')

PLUGIN_PATH = './plugins'
PLUGINS = ['i18n_subsites', 'ipynb.markup', 'tag_cloud', 'tipue_search']
IGNORE_FILES = ['.ipynb_checkpoints']

#TO DO

#ADD DISQUS
##DISQUS_SITENAME = 
#ADD GOOGLE ANALYTICS
#ADD CREATIVE COMMONS
#GOOGLE_ANALYTICS

