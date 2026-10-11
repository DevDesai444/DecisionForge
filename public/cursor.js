/* DecisionForge demo cursor engine — zero dependencies.
   Glides a virtual cursor along an eased, slightly arced path to a target,
   fires a click ripple + press effect, and keeps its position between steps. */
(function () {
  const style = document.createElement("style");
  style.textContent = `
.oj-cursor{position:absolute;left:0;top:0;z-index:99;pointer-events:none;will-change:transform}
.oj-cursor .oj-cursor-inner{display:inline-block;font-size:24px;line-height:1;
  filter:drop-shadow(0 3px 4px rgba(0,0,0,.45));transition:transform .12s ease}
.oj-cursor.oj-pressed .oj-cursor-inner{transform:scale(.7)}
.oj-ripple{position:absolute;width:14px;height:14px;margin:-7px 0 0 -7px;border-radius:50%;
  border:2px solid #4f8ff7;background:rgba(79,143,247,.28);pointer-events:none;z-index:98;
  animation:oj-ripple .6s ease-out forwards}
@keyframes oj-ripple{from{transform:scale(.35);opacity:.95}to{transform:scale(3.4);opacity:0}}`;
  document.head.appendChild(style);

  window.OJCursor = function (container, cursorEl) {
    let cx = null, cy = null;
    const inner = cursorEl.querySelector(".oj-cursor-inner") || cursorEl;
    cursorEl.classList.add("oj-cursor");
    cursorEl.style.transform = "translate(-100px,-100px)"; // offstage until first move

    function place(x, y) { cursorEl.style.transform = `translate(${x}px,${y}px)`; }

    async function glideTo(x, y, duration = 680, arc = 16) {
      if (cx === null) { cx = 26; cy = 18; } // corner entry on first move / after reset
      const x0 = cx, y0 = cy, x1 = x, y1 = y;
      const mx = (x0 + x1) / 2, my = (y0 + y1) / 2;
      const dx = x1 - x0, dy = y1 - y0, len = Math.hypot(dx, dy) || 1;
      const side = (Math.round(mx / 17) + Math.round(my / 11)) % 2 ? 1 : -1;
      const bx = mx - dy / len * arc * side, by = my + dx / len * arc * side;
      const t0 = performance.now();
      await new Promise((resolve) => {
        (function step(now) {
          const t = Math.min(1, (now - t0) / duration);
          const e = t < .5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
          const u = 1 - e;
          place(u * u * x0 + 2 * u * e * bx + e * e * x1,
                u * u * y0 + 2 * u * e * by + e * e * y1);
          if (t < 1) requestAnimationFrame(step); else resolve();
        })(t0);
      });
      cx = x1; cy = y1;
    }

    function press(target) {
      cursorEl.classList.add("oj-pressed");
      setTimeout(() => cursorEl.classList.remove("oj-pressed"), 200);
      if (!target) return;
      target.classList.add("oj-clicked");
      setTimeout(() => target.classList.remove("oj-clicked"), 320);
      const ripple = document.createElement("span");
      ripple.className = "oj-ripple";
      ripple.style.left = cx + "px";
      ripple.style.top = cy + "px";
      container.appendChild(ripple);
      setTimeout(() => ripple.remove(), 700);
    }

    function centerOf(el) {
      const r = el.getBoundingClientRect(), c = container.getBoundingClientRect();
      return { x: r.left - c.left + r.width * 0.74, y: r.top - c.top + r.height * 0.3 };
    }

    return {
      glideTo, press, centerOf,
      async glideToElement(el, duration) {
        if (!el) return;
        const { x, y } = centerOf(el);
        await glideTo(x, y, duration);
      },
      reset() { cx = null; cy = null; cursorEl.style.transform = "translate(-100px,-100px)"; },
    };
  };
})();
