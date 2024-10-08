from django.contrib import admin
from django.urls import path, include
from django.contrib.staticfiles.urls import staticfiles_urlpatterns
from django.views.generic import RedirectView
from django.conf.urls import url

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('requirement.urls')),
    path('retrieval/', include('retrieval.urls')),
    path('extraction/', include('extraction.urls')),
    path('analytic/', include('analytic.urls')),
    path('contact/', include('contact.urls')),
    path('comparison/', include('comparison.urls')),
    path('financial/', include('financial.urls')),

    url(r'^favicon\.ico$',RedirectView.as_view(url='/static/images/favicon.ico')),
]

urlpatterns += staticfiles_urlpatterns()
