// the mode chosen on this device, light, dark or as the system is, set before the page's first
// paint: a script of its own, as the app's content security policy runs no script within a page
document.documentElement.dataset.mode = localStorage.getItem("leat.mode") ?? "system";
