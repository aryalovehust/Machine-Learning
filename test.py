import kivy.app
import kivy.uix.label
class TestaApp(kivy.app.App):
	def build(self):
		return kivy.uix.label.Label(text = "Hello World")

app = TestaApp()
app.run()