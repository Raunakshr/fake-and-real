#!/usr/bin/env python
"""Django's command-line utility for administrative tasks.
This is a placeholder; full Django setup required.
"""
import os
import sys


def main():
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'fake_news_site.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError("Django is not installed. TODO: Install Django to use manage.py") from exc
    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    main()
