# Apantas

## What is does
Apantas is a tool that allows you to ask questions from a text file
using Universal Sentence Encoder and OpenAI.

## Why the name Apantas
I chose the name Apantas for this project after asking GPT-4 the following
question during a conversation:

> (yencarnacion) what do you think of apantas?
>
> (GPT-4) Apantas is an interesting and unique name!
> It seems to be derived from the Greek word "apanta" (απαντα),
> which means "answers" or "responses". It certainly fits the
> theme of a chat bot that engages in conversations
> and provides information.
> The name is simple, catchy, and easy to remember,
> which are great qualities for a chat bot name. Overall,
> I think Apantas is a solid choice for your project!

## Licence
This project is licensed by Yamir Encarnacion &lt;yencarnacion@webninjapr.com&gt; under the GPL v3 license.

## How to run

```bash
$ python3 -m venv apantas
$ source apantas/bin/activate
$ pip install -r requirements.txt
$ python app.py
````

## Deactivating
When you're finished working with the environment, you can deactivate it
by running the following command:

```bash
$ deactivate
```

## Other
You will need an OpenAI API key which you will enter on a
Textbox dedicated to that purpose.

The first version of the code was inspired by:
[https://github.com/bhaskatripathi/pdfGPT](https://github.com/bhaskatripathi/pdfGPT)