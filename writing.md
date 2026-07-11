---
layout: page
title: Writing
description: Notes on machine learning, systems, and the details that become clearer when written down.
permalink: /writing/
---

<div class="writing-list">
  {% for post in site.posts %}
    <article class="writing-item">
      <time datetime="{{ post.date | date_to_xmlschema }}">{{ post.date | date: "%Y" }}</time>
      <h2><a href="{{ post.url | relative_url }}">{{ post.title }}</a></h2>
    </article>
  {% endfor %}
</div>