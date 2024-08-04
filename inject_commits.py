#!/usr/bin/env python
import argparse
import os
import json
from datetime import datetime, timedelta
from subprocess import Popen
import sys

def main(def_args=sys.argv[1:]):
    args = arguments(def_args)
    curr_date = datetime.now()
    directory = 'repository-' + curr_date.strftime('%Y-%m-%d-%H-%M-%S')
    repository = args.repository
    user_name = args.user_name
    user_email = args.user_email
    if repository is not None:
        start = repository.rfind('/') + 1
        end = repository.rfind('.')
        directory = repository[start:end]
    days_before = args.days_before
    if days_before < 0:
        sys.exit('days_before must not be negative')
    days_after = args.days_after
    if days_after < 0:
        sys.exit('days_after must not be negative')
    
    # Check if directory already exists
    if not os.path.exists(directory):
        os.mkdir(directory)
    else:
        print(f"Directory '{directory}' already exists. Using existing directory.")
    
    os.chdir(directory)
    run(['git', 'init', '-b', 'main'])

    if user_name is not None:
        run(['git', 'config', 'user.name', user_name])

    if user_email is not None:
        run(['git', 'config', 'user.email', user_email])

    # Leer commits desde el archivo JSON
    try:
        with open('commits_data.json', 'r') as file:
            commits_data = json.load(file)
    except FileNotFoundError:
        sys.exit("Error: 'commits_data.json' file not found.")

    # Ajustar fechas de los commits
    start_date = datetime(2024, 8, 4)
    end_date = datetime(2024, 10, 20)
    total_days = (end_date - start_date).days
    for i, commit in enumerate(commits_data):
        commit_date = start_date + timedelta(days=i % total_days)
        commit['date'] = commit_date.strftime('%Y-%m-%dT%H:%M:%S+00:00')

    # Generar commits adicionales si es necesario
    while len(commits_data) < 76:
        commit_date = start_date + timedelta(days=len(commits_data) % total_days)
        commits_data.append({
            "sha": "dummy_sha",
            "message": "Fix problems of memory, test, etc.",
            "date": commit_date.strftime('%Y-%m-%dT%H:%M:%S+00:00'),
            "files": [{"filename": "README.md", "changes": 1}]
        })

    # Inyectar commits en el repositorio
    for commit in commits_data:
        contribute(commit)

    if repository is not None:
        run(['git', 'remote', 'add', 'origin', repository])
        run(['git', 'branch', '-M', 'main'])
        run(['git', 'push', '-u', 'origin', 'main'])

    print('\nRepository generation ' +
          '\x1b[6;30;42mcompleted successfully\x1b[0m!')

def contribute(commit):
    date = datetime.strptime(commit['date'], '%Y-%m-%dT%H:%M:%S+00:00')
    with open(os.path.join(os.getcwd(), 'README.md'), 'a') as file:
        file.write(commit['message'] + '\n\n')
    run(['git', 'add', '.'])
    run(['git', 'commit', '-m', commit['message'],
         '--date', date.strftime('"%Y-%m-%d %H:%M:%S"')])

def run(commands):
    Popen(commands).wait()

def arguments(argsval):
    parser = argparse.ArgumentParser()
    parser.add_argument('-nw', '--no_weekends',
                        required=False, action='store_true', default=False,
                        help="""do not commit on weekends""")
    parser.add_argument('-mc', '--max_commits', type=int, default=10,
                        required=False, help="""Defines the maximum amount of
                        commits a day the script can make. Accepts a number
                        from 1 to 20. If N is specified the script commits
                        from 1 to N times a day. The exact number of commits
                        is defined randomly for each day. The default value
                        is 10.""")
    parser.add_argument('-fr', '--frequency', type=int, default=80,
                        required=False, help="""Percentage of days when the
                        script performs commits. If N is specified, the script
                        will commit N%% of days in a year. The default value
                        is 80.""")
    parser.add_argument('-r', '--repository', type=str, required=False,
                        help="""A link on an empty non-initialized remote git
                        repository. If specified, the script pushes the changes
                        to the repository. The link is accepted in SSH or HTTPS
                        format. For example: git@github.com:user/repo.git or
                        https://github.com/user/repo.git""")
    parser.add_argument('-un', '--user_name', type=str, required=False,
                        help="""Overrides user.name git config.
                        If not specified, the global config is used.""")
    parser.add_argument('-ue', '--user_email', type=str, required=False,
                        help="""Overrides user.email git config.
                        If not specified, the global config is used.""")
    parser.add_argument('-db', '--days_before', type=int, default=365,
                        required=False, help="""Specifies the number of days
                        before the current date when the script will start
                        adding commits. For example: if it is set to 30 the
                        first commit date will be the current date minus 30
                        days.""")
    parser.add_argument('-da', '--days_after', type=int, default=0,
                        required=False, help="""Specifies the number of days
                        after the current date until which the script will be
                        adding commits. For example: if it is set to 30 the
                        last commit will be on a future date which is the
                        current date plus 30 days.""")
    return parser.parse_args(argsval)

if __name__ == "__main__":
    main()