#!/usr/bin/env python
# -*- coding: utf-8 -*- #
from __future__ import unicode_literals

AUTHOR = 'Seng Chu'
SITENAME = 'Coding Disciple'
SITEURL = 'http://localhost:8000'
PATH = 'content'
TIMEZONE = 'America/Los_Angeles'
DEFAULT_LANG = 'en'

#General settings
DEFAULT_PAGINATION = 100
DISQUS_SITENAME = 'codingdisciple'

##Pelican-bootstrap3 html changes
#Custom CSS in templates/ipynb.css, injected into base.html
#Edited sidebar title icons, see templates/includes/sidebar
#Injected code into article_list.html for category descriptions
#Edited article_list to remove article summary


#Bootstrap specific settings
THEME = "~pelican-themes/pelican-bootstrap3"
JINJA_ENVIRONMENT = {'extensions': ['jinja2.ext.i18n']}
BOOSTRAP_FLUID = True
BOOTSTRAP_THEME = 'cosmo'
PYGMENTS_STYLE = 'default'
USE_FOLDER_AS_CATEGORY = True

#Social Settings
SOCIAL = (('linkedin', 'https://www.linkedin.com/in/seng-chu-338140142'),
          ('github', 'https://github.com/sengkchu'))

#Bootstrap article settings
DISPLAY_ARTICLE_INFO_ON_INDEX = True
SHOW_SERIES = True
		  
#Bootstrap sidebar tags settings		  
DISPLAY_TAGS_ON_SIDEBAR = True
DISPLAY_TAGS_INLINE = False
TAG_CLOUD_MAX_ITEMS = 10
TAG_CLOUD_STEPS = 6
TAG_CLOUD_SORTING = 'random'
TAG_CLOUD_BADGE = True
TAGS_URL = ''
SIDEBAR_ON_LEFT = True

#Bootstrap sidebar other settings
DISPLAY_ARCHIVE_ON_SIDEBAR = True
DISPLAY_RECENT_POSTS_ON_SIDEBAR = False
MONTH_ARCHIVE_SAVE_AS = '{date:%Y}/{date:%m}/index.html'
RECENT_POST_COUNT = 4
DIRECT_TEMPLATES = ['search']

#Bootstrap banner and brand settings
BANNER = 'images/tree.png'
BANNER_ALL_PAGES = True
BANNER_SUBTITLE = 'A journey of self development'
HIDE_SITENAME = True

#Bootstrap navbar options
DISPLAY_PAGES_ON_MENU = True
DISPLAY_CATEGORIES_ON_MENU = True
MENUITEMS = [('About me', 'http://localhost:8000/')]

#Google Analytics/Add this
GOOGLE_ANALYTICS = 'UA-113145425-1'
ADDTHIS_PROFILE = 'ra-5a6ece335d878db6'
ADDTHIS_DATA_TRACK_ADDRESSBAR = False
#SHARIFF = True
#SHARIFF_ORIENTATION = 'vertical'
#SHARIFF_LANG = 'en'

# Feed generation is usually not desired when developing
FEED_ALL_ATOM = None
CATEGORY_FEED_ATOM = None
TRANSLATION_FEED_ATOM = None
AUTHOR_FEED_ATOM = None
AUTHOR_FEED_RSS = None

MARKUP = ('md', 'ipynb')

PLUGIN_PATH = './plugins'
PLUGINS = ['i18n_subsites', 'ipynb.markup', 'tag_cloud', 'tipue_search', 'series']
IGNORE_FILES = ['.ipynb_checkpoints']

#TO DO
#ADD CREATIVE COMMONS


# Uncomment following line if you want document-relative URLs when developing
#RELATIVE_URLS = True    #Don't know what these do

